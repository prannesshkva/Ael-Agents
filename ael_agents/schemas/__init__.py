# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\xcc\xc7\xd4b\xcfd\x97\x91Z\xc3(BX&\\#'
_P = b'+)PvMgzwzm7GOD%`#sse<=^GB0s09NTLnCl8tyZ?f@Flv<6PGl2oB7Q!h!#+mV`qR0E@}mIdm;PLr{da%je^cM7zzGBM5rcZTGok;A+6zG(o`04f{lv5wF8#3ix8%B?IJQJFldeY`C?s7k|5cUHl$%vcqKfw(nVxsTKA*9aa-F>*ZqZjNtq!;=ZvoRr@9p{xilv^Vxf1%wg1Eq-#|XWW|_ryX#7Kr0HS}M%hcK>9tP!gA|U8s~5r38zKOvxBSO4Ox5E(+NceY&tVM*;R{RSzKT+;)2BA;4H~sBk4^(jTfx_WQIHB&n%Fk)RecfK_L(_n!Da%GCZ!SCO%Ba~ga@)yh!39>xltU5?V@U-z|-9`*Ynw=dHXZO3}jLXF-znENt#*x${bU%;-13aIz%x`yH9y*$UO^~z$B-TNvpi$vOQh`ady!9X2WAxN+5H_L$fo=oMuSGd66ZY<BGOlmOTgzw{LuWi$S8pE*3lb&SRiAgU|PW-$pGGZTl3VzlLGJR*h3hrEEk}#ZNK*ts(~12Pc7O3!NiRl>'
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
