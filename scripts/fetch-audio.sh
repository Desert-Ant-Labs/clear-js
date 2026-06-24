#!/bin/bash
# Mirror the deployed HF Space's audio/ directory into ./audio for local
# dev playback. The deployed Space already has this directory; running
# this is only needed for testing audio playback against serve.py.
#
#   bash clear-web/scripts/fetch-audio.sh

set -e
WEB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASE='https://huggingface.co/spaces/desert-ant-labs/clear-demo/resolve/main/audio'
DEST="$WEB_DIR/audio"

CLIPS=(
  '02-demo_01_-_launch_day__bca8894f-7937-49de-8240-f746'
  '04-portable_video_podcast_setup_001__646677e5-66ec-46'
  '06-the_pitch_03__88508b92-f423-4e65-a973-818ff4002e41'
  '08-wirelessmicreview__be99d613-e272-499d-ad28-38bea7b'
  '09-the_explosion_of_software__c759ed82-38a5-471d-b493'
  '12-foureyes-matt-haig'
  '09-flower-field'
  '07-frank-noise-test'
  '08-randomchats-aries'
  '11-livestream-outdoor'
  '11-randomchats-coachella'
  '12-paul-designing-on-device'
)

for track in raw studio natural; do
  mkdir -p "$DEST/$track"
  for id in "${CLIPS[@]}"; do
    out="$DEST/$track/$id.wav"
    if [ -f "$out" ]; then
      echo "skip   $track/$id.wav"
      continue
    fi
    echo "fetch  $track/$id.wav"
    curl -sL "$BASE/$track/$id.wav" -o "$out"
  done
done

echo
echo "done. served via python3 serve.py"
