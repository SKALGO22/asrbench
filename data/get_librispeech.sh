#!/usr/bin/env bash
# Downloads LibriSpeech test-clean and test-other (~690 MB) into data/LibriSpeech.
set -euo pipefail
cd "$(dirname "$0")"
for s in test-clean test-other; do
  if [ ! -d "LibriSpeech/$s" ]; then
    echo "downloading $s"
    if timeout 1800 wget -q --show-progress -c --timeout=30 "https://www.openslr.org/resources/12/$s.tar.gz" -O "$s.tar.gz" \
       || timeout 1800 wget -q --show-progress -c --timeout=30 "https://us.openslr.org/resources/12/$s.tar.gz" -O "$s.tar.gz"; then
      tar xzf "$s.tar.gz" && rm "$s.tar.gz"
    else
      echo "openslr unreachable; using the Hugging Face copy"; rm -f "$s.tar.gz"
      python get_librispeech_hf.py --subsets "$s"
    fi
  fi
done
echo "done: $(find LibriSpeech -name '*.flac' | wc -l) flac files"
