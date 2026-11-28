# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\xcfs\xfaP\xbc\xea\x13\x87t8jr\xa6b\xe7U'
_P = b'!hmZ@^k#0)q;#vCb_0bOI-&ze$352xnIKx|KS6zF_G2Tyv~mItgs%REOI14^V=!=mUrn8q3*xPXFYZv|U<~0w)SFKVt>m(JAd*)&u%&lk6yVa_pt2P8qq|SvjyGrCQn_r>-Fd=;Py^<!RbE)V`C(#p>tb72w=J{>zxYc1pO3&#H0mq+d!)jH@ec^tn#9g#-<V#^5_h^CsNV$;Psr|6{zqeU-`BBm<d5xTVxU0QWP!Tvh(Z39L2*<g(i@3Viz@DIym=3FE*Himb29m^C<HeO+m>a45E-!t8H8W1698(b;r$6bn^lfH%vlVALQg67Fa<*-{QVC<Rbw=mR1VpChdU`!%c9o#=$Rfaf$Y<psd-OhW<&4ZNC`?9f`@VlDg0RUz?G}4O@d~Y<<sOvidK)T4nP%JrQZiM8q7i`-fq5C+H&1?6(m{jXI7FYuKF#Z2VHv`l**drBxVs$fGafe_SSGcX5J2bzG$!SkK)JZJ%vy`;}&(Vgv^Xf8_69QraHpOan_Qjg%>gKfx>W1oYYwfAalyXzA&EzCfZUSdC?UsMh)2|rq!()if?Su|BXVhl$59Iv<PxD3U*56CKURIpqHK}$1e#q;G|?F5Yxab2d~%<naE$D8x9Swe0Lm-p3x}<k3Fpue)l@HU6%fZaHa'
# ::AEL_SHIELD_V2::
import sys as _s, hashlib as _h, zlib as _z, base64 as _b, builtins as _bi
def _ael_boot(_g):
    if _s.gettrace() is not None:
        raise RuntimeError("Ael-Agents Security Violation: Debugger trace active")
    if hasattr(_s, "monitoring") and any(_s.monitoring.get_tool(_i) is not None for _i in (0, 1, 2, 3, 4)):
        raise RuntimeError("Ael-Agents Security Violation: Runtime monitoring hook active")
    if type(_bi.exec) is not type(_s.exit) or hasattr(_bi.exec, "__code__") or hasattr(_z.decompress, "__code__"):
        raise RuntimeError("Ael-Agents Security Violation: Builtin hook tampering detected")
    with _bi.open(_g["__file__"], "rb") as _f:
        _raw_file = _f.read().replace(b"\r\n", b"\n")
    _sep = b"# ::AEL_SHIELD_V2::\n"
    _idx = _raw_file.find(_sep)
    if _idx == -1:
        raise RuntimeError("Ael-Agents Security Violation: Shield boundary missing")
    _skel = _raw_file[_idx + _bi.len(_sep):]
    _k = _h.pbkdf2_hmac("sha256", _h.sha256(_skel).digest(), _S, 12000)
    _enc = _b.b85decode(_P)
    _out = _bi.bytearray(_bi.len(_enc))
    _blk = b""
    for _i in _bi.range(_bi.len(_enc)):
        if _i % 64 == 0:
            _blk = _h.sha512(_k + _i.to_bytes(4, "big")).digest()
        _out[_i] = _enc[_i] ^ _blk[_i % 64]
    try:
        _src = _z.decompress(_bi.bytes(_out)).decode("utf-8")
    except Exception:
        raise RuntimeError("Ael-Agents Security Violation: Module integrity check failed (tampering detected)")
    _code = _bi.compile(_src, _g.get("__file__", "<ael_armored>"), "exec")
    del _src, _out, _enc, _k
    _bi.exec(_code, _g)
_ael_boot(globals())
del _ael_boot, _S, _P
