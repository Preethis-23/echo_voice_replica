import urllib.request, shutil
from pathlib import Path
urls = [
 'https://github.com/jik876/hifi-gan/releases/download/v0.1/g_025000.pth',
 'https://github.com/jik876/hifi-gan/releases/download/v0.1/g_025000.pt',
 'https://huggingface.co/jik876/hifi-gan/resolve/main/g_025000.pth',
 'https://huggingface.co/jik876/hifi-gan/resolve/main/g_025000.pt',
 'https://huggingface.co/jik876/hifi-gan/resolve/main/generator_v1.pt',
 'https://huggingface.co/jik876/hifi-gan/resolve/main/hifigan_v1.pt',
 'https://huggingface.co/jik876/hifi-gan/resolve/main/g_016000.pth',
 'https://huggingface.co/jik876/hifi-gan/resolve/main/best_g_025000.pth',
 # some forks
 'https://huggingface.co/lj-speech/hifigan/resolve/main/g_025000.pth',
 'https://huggingface.co/lj-speech/hifigan/resolve/main/generator_v1.pt'
]

out_dir = Path('Real-Time-Voice-Cloning') / 'saved_models' / 'default'
out_dir.mkdir(parents=True, exist_ok=True)

for url in urls:
    try:
        print('Trying', url)
        target = out_dir / 'hifigan_try'
        urllib.request.urlretrieve(url, target)
        print('Downloaded to', target, 'size', target.stat().st_size)
        if target.stat().st_size > 1000:
            target2 = out_dir / 'hifigan.pt'
            shutil.move(str(target), str(target2))
            print('Saved as', target2)
            break
    except Exception as e:
        print('Fail', url, e)
print('Done')
