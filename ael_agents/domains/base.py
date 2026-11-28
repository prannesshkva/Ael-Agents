# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'<\xdc\r\xbb\x12\xb3{\xa3\xdb\x1a\xe3\x9b\xedZSB'
_P = b'7Nk2-ukh2B#&IMq0=Pg$+pLNCm~=oUaA{Vr|5^3aftgA0fY^wEHQThGT4fUO@m8m1G=kbEB6+t#3j9<O2fNoMTmx`cVmNPvo2^+w5DT_WD@BbVpvbQnYu!Aog%tBcYu3&bJ$qmNF4oujRs(Xuuga}ARf8drgECzMEOUsNY(+z*?h{^FN!DCda`(z)P(}oN2mc*HG*+TpqghU0yK=oV=LrG3Hm}kGdG>!vAEcxOn_vFC%UWp@<r%=lYb)ziM$M9eM^|c)D;_ELeb8ZQOO4(nLko_hPm%`QgElR{z#i&|o7D2F(86ULI<B35)*FGlKyoeDYr20yHE<lvCGAv?pEZBYg1gD^UQE!X)4TIi$zLd#I5V_KA8cJ~L`9l=HIKNV4zdZ3!~n9Cz0`|3e7Nfm&cIu0L?Z74vxEJk>o)bil1y<GbJ&GMw2X1e*U>I=SS(J+?k|}Z5Pd{BnA410wx3v3qZ|'
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
