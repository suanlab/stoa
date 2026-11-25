Repeat campaigns of notebook BM (2026-11-24/26), each run in a seeded shuffled order so policy and time are
not confounded. campaign.sh reads a run list (one run per line: CPU_GB CONNECTOR POLICY SPEED WORKLOAD) and
runs each through ../run_one.sh into its own directory NN_cpuCPU_GB, so repeats never overwrite each other.
It was run as campaign.sh with campaign1_runs.txt, then with campaign2_runs.txt and campaign3_runs.txt
(the list name and the output directory edited in a copy of the script).
  campaign1_runs.txt  46 runs: traceA600 at 120 GB (both connectors x store-all / oracle at 0.35x and 0.2x,
                      and the causal long-prompt rule at 0.35x), traceA1200 at 40 GB at 0.35x
                      (store-all / store-none) -- random.Random(20261124).shuffle
  campaign2_runs.txt  24 runs: coder600 at 160 GB, and traceA600 with no connector; the 8 offloading runs
                      at 160 GB died at start-up allocating pinned host memory -- random.Random(20261125)
  campaign3_runs.txt  12 runs: coder600 at 120 GB, both connectors x store-all / oracle at 0.12x
