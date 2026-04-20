import torch

from reproducibility.cli_main import get_args
from reproducibility.save_setup import save_setup
from runners.run_main import run_main

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save Setup
    save_setup(args)
    results = run_main(args, device)
    file_path = f"{args.logdir}/{args.diagram_type}_{args.gnn}_{args.seed}"
    if args.diagram_type == "forward_backward":
        if args.fb_one:
            file_path += f"_fbone{int(args.fb_one)}"
        if args.extended_persistence:
            file_path += f"_extpers{int(args.extended_persistence)}"
        if args.extended_persistence2:
            file_path += f"_extpers2{int(args.extended_persistence2)}"
        if args.no_ofst:
            file_path += f"_noofst{int(args.no_ofst)}"
            
    torch.save(
        results, f"{file_path}.results"
    )
    # torch.save(
    #     results, f"{args.logdir}/{args.diagram_type}_{args.gnn}_{args.seed}.results"
    # )
