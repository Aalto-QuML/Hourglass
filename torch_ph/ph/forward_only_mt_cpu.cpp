#include <torch/extension.h>
#include <ATen/ATen.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <limits>
#include <tuple>
#include "unionfindforest.hh"
#pragma omp parallel num_threads(4)

using torch::Tensor;
using torch::indexing::Slice;


// // IMPORTANT: RETURNS local row_id of the dependent row
// inline int64_t ID_dependent_cycle_one_block(
//     torch::TensorAccessor<uint8_t, 2> A, // full (M,M) uint8 accessor
//     int64_t row_offset,                   // e.g. edge_begin
//     int64_t rows_used,                    // number of active rows in this graph
//     int64_t col_offset,                   // e.g. edge_begin
//     int64_t cols_used                     // e.g. edge_end - edge_begin
// ){
//   // basis: pivot_col(local) -> row vector (local coords)
//   std::unordered_map<int64_t, std::vector<uint8_t>> basis;
//   basis.reserve(static_cast<size_t>(rows_used));

//   for (int64_t r_local = 0; r_local < rows_used; ++r_local) {
//     int64_t r = row_offset + r_local;

//     // copy row r (only active columns) into v
//     std::vector<uint8_t> v(static_cast<size_t>(cols_used));
//     int64_t s = 0;
//     for (int64_t j_local = 0; j_local < cols_used; ++j_local) {
//       int64_t j = col_offset + j_local;
//       v[j_local] = A[r][j];
//       s += v[j_local];
//     }
//     if (s == 0) continue; // already zero row

//     // reduce against basis
//     for (auto &kv : basis) {
//       int64_t pivot_local = kv.first;
//       const auto &b = kv.second;
//       if (v[static_cast<size_t>(pivot_local)] != 0) {
//         for (int64_t j_local = 0; j_local < cols_used; ++j_local) {
//           v[j_local] ^= b[static_cast<size_t>(j_local)];
//         }
//       }
//     }

//     // check if reduced to zero
//     s = 0;
//     for (int64_t j_local = 0; j_local < cols_used; ++j_local) s += v[j_local];
//     if (s == 0) return r_local; // return LOCAL row index

//     // otherwise insert with rightmost pivot (local col index)
//     int64_t pivot_local = -1;
//     for (int64_t j_local = cols_used; j_local > 0; --j_local) {
//       if (v[static_cast<size_t>(j_local - 1)] != 0) { pivot_local = j_local - 1; break; }
//     }
//     if (pivot_local >= 0) basis.emplace(pivot_local, std::move(v));
//   }
//   return -1;
// }



template <typename float_t, typename int_t>
void forward_raw(
    torch::TensorAccessor<float_t, 1> filtered_v,   // (n)
    torch::TensorAccessor<float_t, 1> filtered_e,   // (m)
    torch::TensorAccessor<int_t,   2> edge_index,   // (m,2)
    // torch::TensorAccessor<float_t, 1> contracted_v, // (n)
    // torch::TensorAccessor<float_t, 1> contracted_e, // (m)
    // working buffers / state 
    torch::TensorAccessor<int_t,   1> parents,             // (n)
    torch::TensorAccessor<int_t,   2> tree_adj,         // (n,n)
    // torch::TensorAccessor<int_t,   1> depth,               // (n)
    // torch::TensorAccessor<int_t,   1> edge_to_tree_nbr, // (n,n)
    torch::TensorAccessor<int_t,   1> sorting_space,       // (m) indices 0..m-1
    // outputs 
    // torch::TensorAccessor<float_t, 2> persistence0,        // (n,2)
    // torch::TensorAccessor<float_t, 2> persistence1,        // (m,2)
    // outputs: INDICES (not values)
    torch::TensorAccessor<int_t,   2> pers0_idx,        // (n,3): (birth_v, death_edge|-1, death_node|-1)
    torch::TensorAccessor<int_t,   2> pers1_fw_idx,     // (m,2): (birth_edge, death_edge|-1)
    // cycle matrix storage (over-allocated)
    // torch::TensorAccessor<uint8_t, 2> cycle_mat_full,      // (m,m)
    // torch::TensorAccessor<int_t,   1> cycle_birth_edge,    // (m)
    // supernode cycles (over-allocated m x 2), caller will pad/trim
    // torch::TensorAccessor<float_t,2> super_pairs,       // (m,2), init -1
    // torch::TensorAccessor<int_t, 2> super_pairs_idx,   // (m,2)
    // : edge / vertex slice ranges if you batch multiple graphs
    int_t vertex_begin, int_t vertex_end,  // [begin, end)
    int_t edge_begin,   int_t edge_end     // [begin, end)
) {
  const int_t n = vertex_end - vertex_begin;
  const int_t m = edge_end   - edge_begin;

  for (int_t i=vertex_begin;i<vertex_end;++i) {
    // init UF + forest
    // parents[i] = i; tree_adj[i] = -1; depth[i] = 0; edge_to_tree_nbr[i] = -1;
    parents[i] = i; //tree_adj[i] = -1; edge_to_tree_nbr[i] = -1;
    // for (int_t k = 0; k < n; ++k) {
    for (int_t k=vertex_begin;k<vertex_end;++k){   //IMPORTANT: k starts from vertex_begin
      tree_adj[i][k]        = static_cast<int_t>(-1);
      // edge_to_tree_nbr[i][k]= static_cast<int_t>(-1);
    }

    // init outputs
    // persistence0[i][0] = static_cast<float_t>(0);
    // persistence0[i][1] = static_cast<float_t>(0);
    // H0 indices: (birth_v=-1, death_edge=-1, death_node=-1). <<< NOTE ablation no more death node
    pers0_idx[i][0] = static_cast<int_t>(-1);
    pers0_idx[i][1] = static_cast<int_t>(-1);
    // pers0_idx[i][2] = static_cast<int_t>(-1);  <<< NOTE
  }
  for (int_t e=edge_begin;e<edge_end;++e) {
    // persistence1[e][0] = static_cast<float_t>(0);
    // persistence1[e][1] = static_cast<float_t>(0);
    // H1 forward indices: (birth_edge=-1, death_edge=-1)
    pers1_fw_idx[e][0] = static_cast<int_t>(-1);
    pers1_fw_idx[e][1] = static_cast<int_t>(-1);
  }
  // // cycle mats init
  // for (int_t r=0;r<m;++r) {
  //   cycle_birth_edge[r] = static_cast<int_t>(-1);
  //   for (int_t c=0;c<m;++c) cycle_mat_full[r][c] = 0;
  // }
  // // cycle mats init
  // for (int_t r=edge_begin;r<edge_end;++r) { // IMPORTANT: r starts from edge_begin
  //   cycle_birth_edge[r] = static_cast<int_t>(-1);
  //   for (int_t c=edge_begin;c<edge_end;++c) cycle_mat_full[r][c] = 0;
  // }
  // // supernode pairs init (-1)
  // for (int_t r=0;r<m;++r) {
  //   // super_pairs[r][0] = static_cast<float_t>(-1);
  //   // super_pairs[r][1] = static_cast<float_t>(-1);
  //   super_pairs_idx[r][0] = static_cast<int_t>(-1);
  //   super_pairs_idx[r][1] = static_cast<int_t>(-1);
  // }
  // supernode pairs init (-1)
  // for (int_t r=edge_begin;r<edge_end;++r) {
  //   // super_pairs[r][0] = static_cast<float_t>(-1);
  //   // super_pairs[r][1] = static_cast<float_t>(-1);
  //   super_pairs_idx[r][0] = static_cast<int_t>(-1);
  //   super_pairs_idx[r][1] = static_cast<int_t>(-1);
  // }

  // sort edges by filtration using sorting_space
  int_t* sort_beg = sorting_space.data() + edge_begin;
  int_t* sort_end = sorting_space.data() + edge_end;
  std::stable_sort(sort_beg, sort_end, [&filtered_e](int_t i, int_t j){
    return filtered_e[i] < filtered_e[j];
  });

  // FORWARD
  int_t rows_used = 0;  //// <<<< 
  for (int_t k=0;k<m;++k) {
    int_t e_id = sorting_space[edge_begin + k];
    // int_t u = edge_index[0][e_id];
    // int_t v = edge_index[1][e_id];
    int_t u = edge_index[e_id][0];
    int_t v = edge_index[e_id][1];

    int_t cu = UnionFindForest<int_t>::find_label(parents, u);
    int_t cv = UnionFindForest<int_t>::find_label(parents, v);

    if (cu == cv) {
      // chord -> H1 birth
      // persistence1[e_id][0] = filtered_e[e_id];
      // persistence1[e_id][1] = static_cast<float_t>(-1);
      pers1_fw_idx[e_id][0] = e_id;          // birth edge id
      pers1_fw_idx[e_id][1] = static_cast<int_t>(-1); // death later

      // write this cycle in the per-graph row range [edge_begin, edge_begin+rows_used)
      // int_t row_slot = static_cast<int_t>(edge_begin + rows_used);

      // // fundamental cycle: edge + tree path
      // cycle_mat_full[row_slot][e_id] = 1;    //// <<<<
      // auto path = UnionFindForest<int_t>::path_edge_ids(parents, tree_adj, u, v, vertex_begin, vertex_end);
      // for (auto te : path) cycle_mat_full[row_slot][te] = 1;   //// <<<<<
      // cycle_birth_edge[row_slot] = e_id;
      // ++rows_used;
    } else {
      // tree edge: elder rule
      int_t younger = cu, older = cv;
      if (filtered_v[younger] <= filtered_v[older]) {
        std::swap(younger, older); std::swap(u, v);
      }
      // persistence0[younger][0] = filtered_v[younger];
      // persistence0[younger][1] = filtered_e[e_id];
      pers0_idx[younger][0] = younger;        // birth at vertex 'younger'
      pers0_idx[younger][1] = e_id;           // death by this edge
      // pers0_idx[younger][2] = static_cast<int_t>(-1); // not by node
      UnionFindForest<int_t>::merge(parents, tree_adj,
                        u, v, e_id, /*contraction_mode*/false, vertex_begin, vertex_end);   //// <<<< 
    }
  }

  // roots death = -1
  for (int_t i=vertex_begin;i<vertex_end;++i) {
    if (parents[i] == i) {
      // persistence0[i][0] = filtered_v[i];
      // persistence0[i][1] = static_cast<float_t>(-1);
      pers0_idx[i][0] = i;                  // birth at its own vertex
      // pers0_idx[i][1] = -1;              // death_edge stays -1
      // pers0_idx[i][2] = -1;              // death_node stays -1
    }
  }


  // std::cout << "pers0_idx: " <<  std::endl;
  // for (int_t i=vertex_begin;i<vertex_end;++i) {
  //     std::cout << pers0_idx[i][0] << " " << pers0_idx[i][1] << " " << pers0_idx[i][2] << std::endl;
  // }
  // std::cout << "pers1_fw_idx: " <<  std::endl;
  // for (int_t e=edge_begin;e<edge_end;++e) {
  //     std::cout << pers1_fw_idx[e][0] << " " << pers1_fw_idx[e][1] << std::endl;
  // }

  // // BACKWARD
  // // supernode bookkeeping
  // std::vector<uint8_t> in_super(n, 0);    ////<< this is  local  and correct
  // // std::vector<float_t> super_births;
  // std::vector<int_t> super_birth_vertices;    // stack of vertex ids (global ids)


  // int_t super_parent_label = -1;    //// global id 

  // // schedule (time, id, is_node)
  // std::vector<std::tuple<float_t,int_t,bool>> sched;
  // sched.reserve(n + m);   ////<< this is local size and correct 
  // for (int_t i=vertex_begin;i<vertex_end;++i) sched.emplace_back(contracted_v[i], i, true);
  // for (int_t e=edge_begin;  e<edge_end;  ++e) sched.emplace_back(contracted_e[e], e, false);
  // std::sort(sched.begin(), sched.end(),
  //           [](auto& a, auto& b){ return std::get<0>(a) < std::get<0>(b); });


  // // lambda helper (just to be safe as per whiich union find version is used)
  // // auto close_one_infinite_H0 = [&](int_t comp_label, float_t t){
  // //   for (int_t i=vertex_begin;i<vertex_end;++i) {
  // //     if (parents[i] == comp_label) {
  // //       if (persistence0[i][1] == static_cast<float_t>(-1)) {
  // //         persistence0[i][1] = t; break;
  // //       }
  // //     }
  // //   }
  // // };   
  // //// Everything is globally id'd below .
  // auto close_one_infinite_H0_idx = [&](int_t comp_label, int_t node_id){
  //   // std::cout<< "close_one_infinite_H0_idx called with comp_label: " << comp_label << " node_id: " << node_id << std::endl;
  //   for (int_t i=vertex_begin;i<vertex_end;++i) {
  //     if (parents[i] == comp_label) {
  //       // find the open H0 for this component: death_edge==-1 && death_node==-1
  //       if (pers0_idx[i][0] >= 0 && pers0_idx[i][1] == -1 && pers0_idx[i][2] == -1) {
  //         pers0_idx[i][2] = node_id;  // death by contracted_v[node_id]
  //         break;
  //       }
  //     }
  //   }
  // };

  // //// Global perspective below
  // // lambda helper to test alive forward cycles: death == -1
  // auto is_forward_alive = [&](int_t edge_id)->bool {
  //   // return persistence1[edge_id][1] == static_cast<float_t>(-1);
  //   return pers1_fw_idx[edge_id][1] == static_cast<int_t>(-1);
  // };

  // int_t super_pairs_used = 0;

  // for (auto& ev : sched) {
  //   // float_t t = std::get<0>(ev);
  //   int_t    idx = std::get<1>(ev);
  //   bool     is_node = std::get<2>(ev);

  //   if (is_node) {
  //     int_t v_global = idx; // vertex id. but global id
  //     in_super[v_global - vertex_begin] = 1;  // locally id'd boolean flag

  //     if (super_parent_label == -1) {
  //       super_parent_label = parents[v_global];  //correct since local is global node id 
  //     } else {
  //       int_t cu = parents[v_global];  // this is global id of the component of the node which is being merged into the supernode
  //       if (cu != super_parent_label) {
  //         // close_one_infinite_H0(cu, t);
  //         close_one_infinite_H0_idx(cu, v_global);
  //         UnionFindForest<int_t>::merge(parents, tree_adj,
  //                           v_global, static_cast<int_t>(super_parent_label),
  //                           /*edge_id*/-1, /*contraction*/true, vertex_begin, vertex_end);   //// <<<<
  //       }
  //       // super_births.push_back(t); // new supernode cycle birth
  //       super_birth_vertices.push_back(v_global); // new supernode cycle birth . storing global vertex id
  //     }
  //   } else {
  //     // edge contraction
  //     int_t e_id = idx;
  //     // int_t u = edge_index[0][e_id];
  //     // int_t v = edge_index[1][e_id];
  //     int_t u = edge_index[e_id][0];
  //     int_t v = edge_index[e_id][1];
  //     // Ideally should be assert below, but we will ensure correctnes by 
  //     // making sure edge contraction time is >= max of endpoint node contraction times
  //     if (!(in_super[u-vertex_begin] && in_super[v-vertex_begin])) continue;

  //     // Remove this edge from all active cycle rows of THIS graph
  //     for (int_t r_local = 0; r_local < rows_used; ++r_local) {
  //       int_t row_slot = edge_begin + r_local;
  //       cycle_mat_full[row_slot][e_id] = 0;
  //     }
  //     // for (int_t r=0;r<rows_used;++r) cycle_mat_full[r][e_id] = 0;  //// <<<< TODO: fix. edge_being+rows_used  to edge_end

  //     // dependency cull: kill one younger dependent, if exists
  //     {
  //       // pass only the rows that are currently in play: [0..rows_used)
  //       // Tensor active = cycle_mat_full.index({Slice(0, rows_used), Slice()});
  //       const int64_t row_off  = edge_begin;
  //       const int64_t rows     = rows_used;
  //       const int64_t col_off  = edge_begin;
  //       const int64_t cols     = edge_end - edge_begin;

  //       int64_t dep_local = ID_dependent_cycle_one_block(cycle_mat_full, row_off, rows, col_off, cols); ////<<<<< IMP: returns local row id

  //       if (dep_local >= 0) {
  //         int64_t row_slot = row_off + dep_local;          // global row in cycle_mat_full
  //         for (int64_t c = col_off; c < col_off + cols; ++c) cycle_mat_full[row_slot][c] = 0;
  //       }
  //     }

  //     // Any forward cycle killed? (row becomes all-zero)
  //     bool killed_forward = false;
  //     for (int_t r_local = 0; r_local < rows_used; ++r_local) {
  //       int_t row_slot = edge_begin + r_local;
  //       // sum over this row but only across this graph's columns
  //       int64_t s = 0;
  //       for (int_t c = edge_begin; c < edge_end; ++c) s += cycle_mat_full[row_slot][c];
  //       if (s == 0) {
  //         int_t assigned_edge = cycle_birth_edge[row_slot];
  //         if (assigned_edge >= 0 && is_forward_alive(assigned_edge)) {
  //           // forward cycle born at assigned_edge dies at this contraction edge
  //           pers1_fw_idx[assigned_edge][1] = e_id;
  //           killed_forward = true;
  //           break;
  //         }
  //       }
  //     }



  //     if (!killed_forward) {
  //       // kill latest supernode cycle
  //       // again below should assert
  //       // if (!super_births.empty()) {
  //       //   float_t b = super_births.back(); super_births.pop_back();
  //       //   super_pairs[super_pairs_used][0] = b;
  //       //   super_pairs[super_pairs_used][1] = t;
  //       //   ++super_pairs_used;
  //       // }
  //       if (!super_birth_vertices.empty()) {
  //       int_t v_birth = super_birth_vertices.back(); super_birth_vertices.pop_back();
  //       super_pairs_idx[edge_begin + super_pairs_used][0] = v_birth; // from contracted_v[v_birth]
  //       super_pairs_idx[edge_begin + super_pairs_used][1] = e_id;    // dies at contracted_e[e_id]
  //       ++super_pairs_used;
  //       }
  //     }
  //   }
  // }
  // std::cout << "pers0_idx: " <<  std::endl;
  // for (int_t i=vertex_begin;i<vertex_end;++i) {
  //     std::cout << pers0_idx[i][0] << " " << pers0_idx[i][1] << " " << pers0_idx[i][2] << std::endl;
  // }
  // std::cout << "pers1_fw_idx: " <<  std::endl;
  // for (int_t e=edge_begin;e<edge_end;++e) {
  //     std::cout << pers1_fw_idx[e][0] << " " << pers1_fw_idx[e][1] << std::endl;
  // }
}


template <typename float_t, typename int_t>
void forward_batched_ptrs(
    // batched inputs: dims [F, N], [F, M], [2, M], etc.
    torch::TensorAccessor<float_t, 2> filtered_v,
    torch::TensorAccessor<float_t, 2> filtered_e,
    torch::TensorAccessor<int_t,   2> edge_index,
    // torch::TensorAccessor<float_t, 2> contracted_v,
    // torch::TensorAccessor<float_t, 2> contracted_e,
    torch::TensorAccessor<int_t,   1> vertex_slices,  // [G+1]
    torch::TensorAccessor<int_t,   1> edge_slices,    // [G+1]
    // working buffers per filtration (preallocated)
    torch::TensorAccessor<int_t,   2> parents,             // [F, N]
    torch::TensorAccessor<int_t,   3> tree_adj,         // [F, N, N]
    // torch::TensorAccessor<int_t,   2> depth,               // [F, N]
    // torch::TensorAccessor<int_t,   3> edge_to_tree_nbr, // [F, N, N]
    torch::TensorAccessor<int_t,   2> sorting_space,       // [F, M]
    // outputs
    // torch::TensorAccessor<float_t, 3> persistence0,        // [F, N, 2]
    // torch::TensorAccessor<float_t, 3> persistence1,        // [F, M, 2]
    torch::TensorAccessor<int_t,   3> pers0_idx,        // [F, N, 3] (birth_v, death_e|-1, death_v|-1). <<<NOTE ablation no more death_v
    torch::TensorAccessor<int_t,   3> pers1_fw_idx     // [F, M, 2] (birth_e, death_e|-1)
    // torch::TensorAccessor<uint8_t, 3> cycle_mat_full,      // [F, M, M] (or compressed)
    // torch::TensorAccessor<int_t,   2> cycle_birth_edge,    // [F, M]
    // torch::TensorAccessor<float_t, 3> super_pairs          // [F, M, 2]
    // torch::TensorAccessor<int_t, 3> super_pairs_idx       // [F, M, 2]
) {
  auto G = vertex_slices.size(0) - 1;
  auto F = filtered_v.size(0);

  at::parallel_for(0, G * F, 0, [&](int64_t begin, int64_t end) {
    for (auto t = begin; t < end; ++t) {
      int64_t g = t / F; // graph id
      int64_t f = t % F; // filtration id

      auto v_b = vertex_slices[g], v_e = vertex_slices[g+1];
      auto e_b = edge_slices[g],   e_e = edge_slices[g+1];

      forward_raw<float_t,int_t>(
        filtered_v[f], filtered_e[f], edge_index,
        // contracted_v[f], contracted_e[f],
        parents[f], tree_adj[f], //edge_to_tree_nbr[f],
        sorting_space[f],
        // persistence0[f], persistence1[f],
        pers0_idx[f], pers1_fw_idx[f],
        // cycle_mat_full[f], cycle_birth_edge[f], super_pairs_idx[f], //super_pairs[f],
        static_cast<int_t>(v_b), static_cast<int_t>(v_e),
        static_cast<int_t>(e_b), static_cast<int_t>(e_e)
      );
    }
  });
}



// std::tuple<
//     torch::Tensor, // persistence0
//     torch::Tensor, // persistence1
//     torch::Tensor, // cycle_mat_full
//     torch::Tensor, // cycle_birth_edge
//     torch::Tensor  // super_pairs
// >
std::tuple<torch::Tensor, torch::Tensor>
compute_forward_batched_mt(
    torch::Tensor filtered_v,   // [F,N]
    torch::Tensor filtered_e,   // [F,M]
    torch::Tensor edge_index,   // [M,2]
    // torch::Tensor contracted_v, // [F,N]
    // torch::Tensor contracted_e, // [F,M]
    torch::Tensor vertex_slices, // [G+1]
    torch::Tensor edge_slices    // [G+1]
) {
  auto n_filtrations = filtered_v.size(0);
  auto n_nodes = filtered_v.size(1);
  auto n_edges = filtered_e.size(1);

  auto opts_int = edge_index.options().requires_grad(false);
  auto opts_float = filtered_v.options();//.dtype(filt-ered_v.scalar_type()).requires_grad(false);

  // Allocate outputs
  // auto persistence0   = torch::full({n_filtrations, n_nodes, 2}, -1, opts_float);
  // auto persistence1   = torch::full({n_filtrations, n_edges, 2}, -1, opts_float);
  // auto super_pairs    = torch::full({n_filtrations, n_edges, 2}, -1, opts_float);

  auto pers0_idx       = torch::full({n_filtrations, n_nodes, 2}, -1, opts_int);  //<<< NOTE ablation no more death_v
  auto pers1_fw_idx    = torch::full({n_filtrations, n_edges, 2}, -1, opts_int);
  // auto super_pairs_idx = torch::full({n_filtrations, n_edges, 2}, -1, opts_int);
  // auto cycle_mat_full = torch::zeros({n_filtrations, n_edges, n_edges}, torch::dtype(torch::kUInt8).device(filtered_v.device()));
  // auto cycle_birth_edge = torch::full({n_filtrations, n_edges}, -1, opts_int);

  // Working buffers
  auto parents             = torch::arange(n_nodes, opts_int).unsqueeze(0).repeat({n_filtrations,1});
  // auto tree_adj         = torch::full({n_filtrations, n_nodes}, -1, opts_int);
  auto tree_adj        = torch::full({n_filtrations, n_nodes, n_nodes}, -1, opts_int);
  // auto depth               = torch::zeros({n_filtrations, n_nodes}, opts_int);
  // auto edge_to_tree_nbr = torch::full({n_filtrations, n_nodes}, -1, opts_int);
  // auto edge_to_tree_nbr= torch::full({n_filtrations, n_nodes, n_nodes}, -1, opts_int);
  auto sorting_space       = torch::arange(n_edges, opts_int).unsqueeze(0).repeat({n_filtrations,1});

  // Dispatch
  AT_DISPATCH_FLOATING_TYPES(filtered_v.scalar_type(), "compute_forward_mt", ([&] {
    using float_t = scalar_t;
    AT_DISPATCH_INTEGRAL_TYPES(edge_index.scalar_type(), "compute_forward_mt2", ([&] {
      using int_t = scalar_t;

      forward_batched_ptrs<float_t,int_t>(
          filtered_v.accessor<float_t,2>(),
          filtered_e.accessor<float_t,2>(),
          edge_index.accessor<int_t,2>(),
          // contracted_v.accessor<float_t,2>(),
          // contracted_e.accessor<float_t,2>(),
          vertex_slices.accessor<int_t,1>(),
          edge_slices.accessor<int_t,1>(),
          parents.accessor<int_t,2>(),
          tree_adj.accessor<int_t,3>(),
          // depth.accessor<int_t,2>(),
          // edge_to_tree_nbr.accessor<int_t,2>(),
          sorting_space.accessor<int_t,2>(),
          // persistence0.accessor<float_t,3>(),
          // persistence1.accessor<float_t,3>(),
          // cycle_mat_full.accessor<uint8_t,3>(),
          // cycle_birth_edge.accessor<int_t,2>(),
          // super_pairs.accessor<float_t,3>()
          pers0_idx.accessor<int_t,3>(),
          pers1_fw_idx.accessor<int_t,3>()
          // cycle_mat_full.accessor<uint8_t,3>(),
          // cycle_birth_edge.accessor<int_t,2>(),
          // super_pairs_idx.accessor<int_t,3>()
      );
    }));
  }));


  // =============== Gather values back ===================

  // Augmented tensors for safe indexing (extra column for -1 cases)
  auto cv_aug = torch::cat({filtered_v,
                            torch::zeros({n_filtrations,1}, opts_float)}, 1);
  auto ce_aug = torch::cat({filtered_e,
                            torch::zeros({n_filtrations,1}, opts_float)}, 1);
  // auto cv_aug_c = torch::cat({contracted_v,
  //                             torch::zeros({n_filtrations,1}, opts_float)}, 1);
  // auto ce_aug_c = torch::cat({contracted_e,
  //                             torch::zeros({n_filtrations,1}, opts_float)}, 1);

  // -------- H0 --------
  auto b0  = pers0_idx.index({Slice(), Slice(), 0});  // birth due to node filtration. Has to be >=0

  auto de0 = pers0_idx.index({Slice(), Slice(), 1});  // death due to edge filtration. Can be -1
  // auto dv0 = pers0_idx.index({Slice(), Slice(), 2});  // death due to node contraction. Can be -1

  auto de0_cl = torch::where(de0.lt(0), torch::full_like(de0, n_edges), de0);
  // auto dv0_cl = torch::where(dv0.lt(0), torch::full_like(dv0, n_nodes), dv0);


  auto births0 = cv_aug.gather(1, b0);        // Component birth at node filtration
  auto deaths0_e = ce_aug.gather(1, de0_cl);  // Component death due to edge filtration
  // auto deaths0_v = cv_aug_c.gather(1, dv0_cl);  // Component death due to node contraction . NOTE. collect from v_contraction 
  // auto deaths0   = torch::max(deaths0_e, deaths0_v);
  auto persistence0 = torch::stack({births0, deaths0_e}, 2); // [F,N,2]



  // -------- H1 forward --------
  auto b1_fw = pers1_fw_idx.index({Slice(), Slice(), 0});   // birth at edge filtration. initialized with -1. can be -1 if untouched
  auto d1_fw = pers1_fw_idx.index({Slice(), Slice(), 1});   // death at edge filtration. can be -1 if untouched or undead
  auto b1_fw_cl = torch::where(b1_fw.lt(0), torch::full_like(b1_fw, n_edges), b1_fw);
  // auto d1_fw_cl = torch::where(d1_fw.lt(0), torch::full_like(d1_fw, n_edges), d1_fw);

  // std::cout << "b1_fw: " << b1_fw << std::endl;
  // std::cout << "b1_fw_cl: " << b1_fw_cl << std::endl;
  // std::cout << "d1_fw_cl: " << d1_fw_cl << std::endl;

  auto births1_fw = ce_aug.gather(1, b1_fw_cl);     // Cycle birth at edge filtration
  auto deaths1_fw = torch::full_like(births1_fw, -1.0); // infinite / unknown death
  // auto deaths1_fw = ce_aug_c.gather(1, d1_fw_cl);  // (Forward) Cycle death at edge contraction
  auto pers1_fw_vals = torch::stack({births1_fw, deaths1_fw}, 2);

  // // -------- H1 supernode --------
  // auto b1_sp = super_pairs_idx.index({Slice(), Slice(), 0}); // Supernode Cycle birth at node contraction
  // auto d1_sp = super_pairs_idx.index({Slice(), Slice(), 1}); // Supernode Cycle death at edge contraction
  // auto b1_sp_cl = torch::where(b1_sp.lt(0), torch::full_like(b1_sp, n_nodes), b1_sp);
  // auto d1_sp_cl = torch::where(d1_sp.lt(0), torch::full_like(d1_sp, n_edges), d1_sp);

  // auto births1_sp = cv_aug_c.gather(1, b1_sp_cl);   // Cycle birth at node contraction 
  // auto deaths1_sp = ce_aug_c.gather(1, d1_sp_cl);   // Cycle death at edge contraction
  // auto pers1_sp_vals = torch::stack({births1_sp, deaths1_sp}, 2);

  // -------- Combine H1 --------
  // auto persistence1 = torch::cat({pers1_fw_vals, pers1_sp_vals}, 1); // [F,2M,2]

  // return std::make_tuple(std::move(persistence0), std::move(persistence1));
  return std::make_tuple(std::move(persistence0), std::move(pers1_fw_vals));

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_forward_batched_mt",
        &compute_forward_batched_mt,
        py::call_guard<py::gil_scoped_release>(),
        "Forward–only persistence routine (multi-threaded)");
}
