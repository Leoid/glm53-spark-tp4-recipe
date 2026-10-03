set -x
CU=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
for h in $CU/*.h; do b=$(basename $h); [ -e /usr/local/cuda/include/$b ] || ln -s $h /usr/local/cuda/include/$b; done
DP=/usr/local/lib/python3.12/dist-packages
pip uninstall -y exllamav3 2>/dev/null || true
rm -rf $DP/exllamav3 $DP/exllamav3-*.dist-info $DP/exllamav3_ext*.so
cd /work/exllamav3
unset EXLLAMA_NOCOMPILE
export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=20
pip install --force-reinstall --no-deps --no-build-isolation -v . 2>&1 | tail -30
echo BUILD_EXIT=${PIPESTATUS[0]}
ls -la $DP/exllamav3_ext*.so
python3 -c "import importlib.metadata as m; print(\"INSTALLED_VERSION\", m.version(\"exllamav3\"))"
python3 -c "from exllamav3.architecture.glm_moe_dsa import GlmMoeDsaConfig; print(\"ARCH_OK\")"
echo BUILD_DONE
