#!/bin/bash
# Host tuning for the AI edit / ✨ create features (Tesla P4). Run: sudo bash ops/ai-host-tune.sh
#
# A. ComfyUI: --use-pytorch-cross-attention instead of --use-split-cross-attention (fp8 weights
#    kept). Measured 2026-10-28 on the P4, SDXL 768x1024: 6.0 vs 6.8 s/step (cfg 5.5), model
#    load 19 vs 32 s. fp16 weights were slower (6.9) and split+fp16 slowest (8.3).
#    Installed as a drop-in (comfyui.service.d/booru-sdpa.conf), the original unit is untouched.
#
# B. Keep Ollama on the GPU instead of silently falling back to the CPU.
#
# Why B: at boot Ollama's GPU discovery has a fixed 30 s watchdog; on this machine the first
# CUDA init after a boot takes longer (18 s even on a warm system), so Ollama logs
# "llama-server GPU discovery watchdog timed out" and runs CPU-only (50-110 s per AI-edit
# prompt instead of ~5-10 s) until someone restarts it.
#
# What this installs (run as root: sudo bash ops/ollama-gpu-fix.sh):
#   1. drop-in ollama.service.d/booru-gpu.conf: a big CUDA JIT cache (so kernels compiled
#      once stay compiled) and no Vulkan probing (cuda_v12 is already pinned).
#   2. /usr/local/sbin/ollama-gpu-check + ollama-gpu-check.service/.timer: 2 min after boot
#      (and every 30 min) it looks at the CURRENT Ollama run's "inference compute" log
#      line; if it says library=cpu while nvidia-smi sees the GPU, Ollama is restarted
#      (up to 3 tries per check). By then the GPU is warm and discovery succeeds.
# Undo: sudo rm /etc/systemd/system/comfyui.service.d/booru-sdpa.conf \
#         /etc/systemd/system/ollama.service.d/booru-gpu.conf \
#         /etc/systemd/system/ollama-gpu-check.{service,timer} /usr/local/sbin/ollama-gpu-check
#       && sudo systemctl daemon-reload && sudo systemctl restart ollama comfyui
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

# ── A. ComfyUI attention ──
install -d /etc/systemd/system/comfyui.service.d
cat > /etc/systemd/system/comfyui.service.d/booru-sdpa.conf <<'EOF'
[Service]
ExecStart=
ExecStart=/home/equa/git/ComfyUI/venv/bin/python main.py --listen 0.0.0.0 --port 8188 --disable-dynamic-vram --cuda-device 0 --fp8_e4m3fn-unet --use-pytorch-cross-attention --output-directory /@/Media_SRV/Photo
EOF

# ── B. Ollama on the GPU ──
install -d /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/booru-gpu.conf <<'EOF'
[Service]
# keep compiled CUDA kernels (sm_61 JIT) between restarts; default cache is only 256 MiB
Environment="CUDA_CACHE_MAXSIZE=4294967296"
# cuda_v12 is pinned (OLLAMA_LLM_LIBRARY); probing Vulkan too only adds discovery time
Environment="OLLAMA_VULKAN=0"
EOF

cat > /usr/local/sbin/ollama-gpu-check <<'EOF'
#!/bin/bash
# Restart Ollama when its current run landed on the CPU although the GPU is there.
nvidia-smi -L >/dev/null 2>&1 || exit 0              # no GPU visible: nothing to fix
for try in 1 2 3; do
    for i in $(seq 1 30); do                          # wait for this run's discovery result
        inv=$(systemctl show -p InvocationID --value ollama)
        line=$(journalctl -o cat _SYSTEMD_INVOCATION_ID="$inv" 2>/dev/null | grep 'msg="inference compute"' | tail -1)
        [ -n "$line" ] && break
        sleep 5
    done
    case "$line" in
        *library=CUDA*) echo "ollama is on the GPU"; exit 0 ;;
        *library=cpu*)  echo "ollama landed on the CPU (try $try), restarting"; systemctl restart ollama ;;
        *)              echo "no discovery result yet"; exit 0 ;;
    esac
done
EOF
chmod 755 /usr/local/sbin/ollama-gpu-check

cat > /etc/systemd/system/ollama-gpu-check.service <<'EOF'
[Unit]
Description=Restart Ollama if it fell back to the CPU
After=ollama.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/ollama-gpu-check
EOF

cat > /etc/systemd/system/ollama-gpu-check.timer <<'EOF'
[Unit]
Description=Check that Ollama runs on the GPU (after boot, then every 30 min)

[Timer]
OnBootSec=2min
OnUnitActiveSec=30min

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl restart comfyui
systemctl enable --now ollama-gpu-check.timer
systemctl restart ollama
/usr/local/sbin/ollama-gpu-check
