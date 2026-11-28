# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\x1bEN\xf16\x88\xd2\xf8\xec\x0e.r\xd7\x91t\x1f'
_P = b'n)l|JbKpFN;t@BPR+sM!5tSzjbt-oNpEi(5x6p1yw}D(a45JL?fpyJz{RjCK<NLN39-6as9R0p~P6z*fHw$sC>+8pQCW@MIwE}7+u^5j`?010<f(^>z0#-DeV&tTDiI#69aQslZ8j^7R80ENa52X5P551Rr!}QkE^eyEl&M5+PyO5dY{~-acss&i2edxCf@`_{HCar2AIh->R*~yy~ZTc|H$eM}kE0JUrck4PGIKPdb+!Mg3zqfIwX!~ka2>D2)S`{ka(NHg?<Yao2X}oM7eBgEO`QW_XT!$EHRw>05c1pMLGX'
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
