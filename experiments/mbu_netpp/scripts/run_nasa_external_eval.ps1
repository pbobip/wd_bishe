param(
    [string]$PreparedRoot = "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_nasa_super_all",
    [string]$OutputRoot = "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\outputs\external_eval",
    [string[]]$Experiments = @(
        "e1_unetpp_gpu",
        "e1_micronet_unetpp_gpu",
        "e2_micronet_edge_deep_gpu",
        "e3_micronet_edge_deep_vf_gpu"
    ),
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$repoRoot = Split-Path -Parent $PSScriptRoot

foreach ($exp in $Experiments) {
    $checkpointRoot = Join-Path $repoRoot "outputs\$exp"
    foreach ($fold in 0..2) {
        $checkpoint = Join-Path $checkpointRoot ("fold_{0}\best.pt" -f $fold)
        $outdir = Join-Path $OutputRoot "$exp\nasa_super_all\fold_$fold"
        python -m experiments.mbu_netpp.external_eval `
            --checkpoint $checkpoint `
            --prepared-root $PreparedRoot `
            --output-dir $outdir `
            --device $Device `
            --split test different_test
    }
}
