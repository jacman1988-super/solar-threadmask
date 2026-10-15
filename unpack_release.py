EXPECTED_SHA256='10852010996fa159a53bfc8b00696edec0dab07a9b22edbb3d10c350be91f7ca'
"""Expand the approved public release after verifying archive bytes and paths."""
import hashlib,json,zipfile
from pathlib import Path,PurePosixPath
root=Path(__file__).resolve().parent
archive=root/'source-and-results.zip'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=EXPECTED_SHA256:raise ValueError('Archive checksum mismatch')
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        path=PurePosixPath(item.filename)
        if path.is_absolute() or '..' in path.parts or ':' in item.filename or '\\' in item.filename or (item.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe archive path')
        if not (root.joinpath(*path.parts).resolve().is_relative_to(root.resolve())):raise ValueError('Path outside checkout')
    if z.testzip() is not None:raise ValueError('Archive CRC failure')
    z.extractall(root)
for name,digest in json.loads((root/'manifest.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Expanded file mismatch: '+name)
print('Complete source/results release expanded and verified. Follow README instructions.')
