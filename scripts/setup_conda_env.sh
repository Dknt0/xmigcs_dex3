#!/usr/bin/env bash
# Install the Tienkung environment hook + tiensim launcher into the ACTIVE
# conda env. Run after `conda activate xhumanoid`:
#   bash scripts/setup_conda_env.sh
# Idempotent — safe to re-run (also after creating a fresh env).
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "error: no active conda env (run: conda activate xhumanoid)" >&2
  exit 1
fi

PYVER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYVER" != "3.12" ]; then
  echo "error: expected a Python 3.12 env (rclpy/jazzy + cp312 sptlib wheel), got $PYVER" >&2
  exit 1
fi

ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACT_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACT_DIR" "$DEACT_DIR"

# --- activation hook: ROS, ROS_DOMAIN_ID, auto-detected DISPLAY/XAUTHORITY ---
cat > "$ACT_DIR/tienkung-env.sh" <<'EOF'
# Tienkung xhumanoid env — sourced by conda on every activation.
# Installed by Deploy_Tienkung/scripts/setup_conda_env.sh

# ROS 2 Jazzy (must be sourced AFTER conda activation; zsh vs bash)
if [ -n "${ZSH_VERSION:-}" ]; then
  . /opt/ros/jazzy/setup.zsh
else
  . /opt/ros/jazzy/setup.bash
fi

export ROS_DOMAIN_ID="${TIENKUNG_ROS_DOMAIN_ID:-42}"

# Auto-detect a working X display for the MuJoCo viewer (never hardcode:
# Wayland+Xwayland random auth suffix, GDM :1 comes and goes).
_TK_PREV_DISPLAY="${DISPLAY:-}"
_TK_PREV_XAUTH="${XAUTHORITY:-}"
_TK_PID="$(pgrep -u "$(id -u)" -x nvim | head -1)"
if [ -n "$_TK_PID" ]; then
  # 1. inherit from a running GUI app
  DISPLAY="$(tr '\0' '\n' < "/proc/$_TK_PID/environ" | sed -n 's/^DISPLAY=//p')"
  XAUTHORITY="$(tr '\0' '\n' < "/proc/$_TK_PID/environ" | sed -n 's/^XAUTHORITY=//p')"
fi
if [ -z "${DISPLAY:-}" ] || ! DISPLAY="$DISPLAY" XAUTHORITY="${XAUTHORITY:-}" xdpyinfo >/dev/null 2>&1; then
  # 2. probe display x auth cross-product
  _TK_DISP=""; _TK_AUTH=""
  for _TK_D in :0 :1; do
    for _TK_A in "/run/user/$(id -u)"/.mutter-Xwaylandauth.* "/run/user/$(id -u)/gdm/Xauthority"; do
      [ -f "$_TK_A" ] || continue
      if DISPLAY="$_TK_D" XAUTHORITY="$_TK_A" xdpyinfo >/dev/null 2>&1; then
        _TK_DISP="$_TK_D"; _TK_AUTH="$_TK_A"; break 2
      fi
    done
  done
  if [ -n "$_TK_DISP" ]; then
    DISPLAY="$_TK_DISP"; XAUTHORITY="$_TK_AUTH"
  else
    # keep whatever was there before, else last resort
    DISPLAY="${_TK_PREV_DISPLAY:-:0}"
    XAUTHORITY="${_TK_PREV_XAUTH:-$HOME/.Xauthority}"
  fi
fi
export DISPLAY XAUTHORITY
unset _TK_PID _TK_DISP _TK_AUTH _TK_D _TK_A
EOF

# --- deactivation hook: restore pre-activation values ---
cat > "$DEACT_DIR/tienkung-env.sh" <<'EOF'
# Tienkung xhumanoid env — restore pre-activation DISPLAY/XAUTHORITY.
if [ -n "${_TK_PREV_DISPLAY+x}" ]; then DISPLAY="$_TK_PREV_DISPLAY"; else unset DISPLAY; fi
if [ -n "${_TK_PREV_XAUTH+x}" ]; then XAUTHORITY="$_TK_PREV_XAUTH"; else unset XAUTHORITY; fi
export DISPLAY XAUTHORITY 2>/dev/null || true
unset ROS_DOMAIN_ID _TK_PREV_DISPLAY _TK_PREV_XAUTH
EOF

install -m 0755 "$(dirname "$0")/tiensim.sh" "$CONDA_PREFIX/bin/tiensim"

echo "installed:"
echo "  $ACT_DIR/tienkung-env.sh"
echo "  $DEACT_DIR/tienkung-env.sh"
echo "  $CONDA_PREFIX/bin/tiensim"
