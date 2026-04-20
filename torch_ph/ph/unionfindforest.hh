#pragma once
#include <ATen/ATen.h>
#include <vector>
#include <stdexcept>
#include <queue>


template <typename int_type> class UnionFindForest {
public:

  static inline int_type find_label(at::TensorAccessor<int_type, 1> parents, int_type u) {
    if (u < 0 || u >= parents.size(0)) {
      throw std::out_of_range("find_label: u out of bounds");
    }
    return parents[u]; 
  }

  static inline bool connected(at::TensorAccessor<int_type, 1> parents,
                               int_type u, int_type v) {
    return find_label(parents, u) == find_label(parents, v);
  }

  // merge + optional forest maintenance 
  static inline void merge(at::TensorAccessor<int_type, 1> parents,
                           at::TensorAccessor<int_type, 2> tree_adj,
                           int_type p, int_type q, int_type edge_id,
                           bool contraction_mode,
                           int_type vertex_begin, int_type vertex_end) {
    const int64_t N = parents.size(0);
    if (p < 0 || p >= N || q < 0 || q >= N) {
      throw std::out_of_range("merge: p/q out of bounds");
    }

    // p,q must belong to the current graph slice
    if (!(p >= vertex_begin && p < vertex_end &&
          q >= vertex_begin && q < vertex_end)) {
      throw std::runtime_error("merge: p/q not in current vertex slice");
    }

    int_type root_p = parents[p];
    int_type root_q = parents[q];
    if (root_p == root_q) return; // already same component

    // Record the spanning-tree edge in the single adjacency buffer (if not contracting)
    if (!contraction_mode) {
      tree_adj[p][q] = edge_id;
      tree_adj[q][p] = edge_id;
    }

    // O(N) relabel: all vertices with label root_p become label root_q
    // for (int64_t i = 0; i < N; ++i) {
    // O(|slice|) relabel inside this graph only
    for (int_type i = vertex_begin; i < vertex_end; ++i) {
      if (parents[i] == root_p) {
        parents[i] = root_q;
      }
    }
  }



  static inline std::vector<int_type>
  path_edge_ids(at::TensorAccessor<int_type, 1> parents,
                at::TensorAccessor<int_type, 2> tree_adj,
                int_type u, int_type v,
                int_type vertex_begin, int_type vertex_end) {

    const int64_t N = parents.size(0);
    if (u < 0 || u >= N || v < 0 || v >= N)
      throw std::out_of_range("path_edge_ids: u/v out of bounds");
    if (!(u >= vertex_begin && u < vertex_end &&
          v >= vertex_begin && v < vertex_end)) {
      throw std::runtime_error("path_edge_ids: u/v not in current vertex slice");
    }
    if (!connected(parents, u, v))
      throw std::runtime_error("path_edge_ids: u and v must be connected");


    // BFS within the slice
    const int_type SL = vertex_end - vertex_begin;
    std::vector<int_type> prev_node(SL, static_cast<int_type>(-1));
    std::vector<int_type> prev_edge(SL, static_cast<int_type>(-1));

    auto to_local = [&](int_type g) -> int_type { return g - vertex_begin; };
    auto to_global = [&](int_type l) -> int_type { return l + vertex_begin; };


    // std::vector<int_type> prev_node(N, static_cast<int_type>(-1));
    // std::vector<int_type> prev_edge(N, static_cast<int_type>(-1));
    std::queue<int_type> q;
    int_type uL = to_local(u), vL = to_local(v);
    prev_node[uL] = uL;    // mark visited and root of BFS tree
    q.push(uL);


    while (!q.empty()) {
      int_type xL = q.front(); q.pop();
      if (xL == vL) break;
      int_type xG = to_global(xL);

      // scan row x: every k with tree_adj[x][k] != -1 is a neighbor via edge tree_adj[x][k]
      // scan only inside the slice
      for (int_type kG = vertex_begin; kG < vertex_end; ++kG) {
        int_type eid = tree_adj[xG][kG];
        if (eid == static_cast<int_type>(-1)) continue;     // no tree edge
        int_type kL = to_local(kG);
        if (prev_node[kL] != static_cast<int_type>(-1)) continue; // visited

        prev_node[kL] = xL;
        prev_edge[kL] = eid;    // edge xG--kG
        q.push(kL);
        if (kL == vL) { xL = vL; break; }            // optional early-exit
      }
    }

    if (prev_node[vL] == static_cast<int_type>(-1)) {
      throw std::runtime_error("path_edge_ids: path not found in slice (broken forest? or adjacency inconsistent?)");
    }

    // backtrack path. nodes are local indices, edges are global ids
    std::vector<int_type> path;
    for (int_type cur = vL; cur != uL; cur = prev_node[cur]) {
      int_type eid = prev_edge[cur];
      if (eid == static_cast<int_type>(-1)) {
        throw std::runtime_error("path_edge_ids: missing edge id during backtrack");
      }
      path.push_back(eid);
    }

    return path;
  }

};
