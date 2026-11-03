"""Download this private repository's pinned Release artifacts and verify SHA256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def get_credential(repository):
    if shutil.which('gh'):
        p = subprocess.run(['gh', 'auth', 'token', '--hostname', 'github.com'],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    p = subprocess.run(['git', 'credential', 'fill'],
        input='protocol=https\nhost=github.com\npath=' + repository + '.git\n\n',
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=dict(os.environ, GIT_TERMINAL_PROMPT='0'), timeout=20)
    fields = dict(line.split('=', 1) for line in p.stdout.splitlines() if '=' in line)
    if p.returncode or not fields.get('password'):
        raise RuntimeError('No GitHub credential with private-repository read access. Run gh auth login, or configure your Git HTTPS credential helper.')
    return fields['password']


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise RuntimeError('Release download must use HTTPS')
        target = super().redirect_request(req, fp, code, msg, headers, newurl)
        if target and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            target.remove_header('Authorization')
        return target


def open_api(path, token, accept):
    url = 'https://api.github.com' + path
    req = urllib.request.Request(url, headers={
        'Authorization': 'Bearer ' + token, 'Accept': accept,
        'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'supermimic-release-fetch'})
    try:
        return urllib.request.build_opener(SafeRedirect).open(req, timeout=120)
    except urllib.error.HTTPError as exc:
        raise RuntimeError('GitHub API status ' + str(exc.code) + '; check private-repository access and release tag') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['cuboi', 'zita', 'all'], default='all')
    parser.add_argument('--kind', choices=['checkpoint', 'onnx', 'all'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-only', action='store_true', help='Offline checksum check; no GitHub login needed')
    parser.add_argument('--release-id', type=int, help='Maintainer QA: use this Release ID; pinned tag and checksums still apply')
    parser.add_argument('--allow-draft', action='store_true', help='Maintainer QA: permit an authenticated draft readback')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'artifacts/weights-manifest.json').read_text())
    assets = [a for a in manifest['assets']
              if (args.method == 'all' or a['backend'] == args.method)
              and (args.kind == 'all' or a['kind'] == args.kind)]
    if not assets:
        raise RuntimeError('No matching pinned assets')
    if args.dry_run:
        print(json.dumps({'repository': manifest['repository'], 'tag': manifest['release_tag'],
                          'assets': assets, 'output': str(args.output)}, indent=2))
        return
    token, release = None, None
    if not args.verify_only:
        token = get_credential(manifest['repository'])
        release_path = '/repos/' + manifest['repository'] + ('/releases/' + str(args.release_id) if args.release_id else '/releases/tags/' + manifest['release_tag'])
        with open_api(release_path,
                      token, 'application/vnd.github+json') as response:
            release = json.load(response)
        if release['tag_name'] != manifest['release_tag'] or (release['draft'] and not args.allow_draft):
            raise RuntimeError('Pinned Release is not published')
        args.output.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        name = asset['name']
        if Path(name).name != name or len(asset['sha256']) != 64:
            raise RuntimeError('Invalid artifact manifest')
        final = args.output / name
        if final.exists():
            if final.stat().st_size != asset['bytes'] or sha256(final) != asset['sha256']:
                raise RuntimeError('Existing artifact checksum differs; move it aside: ' + str(final))
            print('verified ' + name, flush=True)
            continue
        if args.verify_only:
            raise FileNotFoundError(final)
        match = [a for a in release['assets'] if a['name'] == name]
        if len(match) != 1 or match[0]['size'] != asset['bytes']:
            raise RuntimeError('Release asset missing or size changed: ' + name)
        entry = match[0]
        if entry.get('digest') and entry['digest'] != 'sha256:' + asset['sha256']:
            raise RuntimeError('GitHub digest differs from repository pin: ' + name)
        temp_name = None
        try:
            h, size, last_progress = hashlib.sha256(), 0, 0
            with tempfile.NamedTemporaryFile(prefix=name + '.', suffix='.part', dir=args.output, delete=False) as target:
                temp_name = Path(target.name)
                with open_api('/repos/' + manifest['repository'] + '/releases/assets/' + str(entry['id']),
                              token, 'application/octet-stream') as source:
                    for block in iter(lambda: source.read(4 << 20), b''):
                        target.write(block); h.update(block); size += len(block)
                        if size - last_progress >= 16 << 20:
                            print('downloading ' + name + ' ' + str(size) + '/' + str(asset['bytes']), flush=True)
                            last_progress = size
            if size != asset['bytes'] or h.hexdigest() != asset['sha256']:
                raise RuntimeError('Downloaded artifact checksum differs: ' + name)
            os.replace(temp_name, final)
            temp_name = None
            print('downloaded and verified ' + name, flush=True)
        finally:
            if temp_name is not None:
                temp_name.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
