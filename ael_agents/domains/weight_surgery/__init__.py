# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\xa7\xde\xc10\x0f\xf9\xae\xcc\xcd\xbevZw\x01/\xad'
_P = b'M3)~E3^qx-iC*5QQz2c-pDM18zR;|xxIu7{tGM)i|0*EC_9xoR#+iH1<X9h@6D4`bEDj=qHfmG^>r=B8fjr|B9wP;|EwA;yox)CeQeH6FvI$GipcO5DJlSf*jzxqVK*r30-KenfVhhXdb3XoS=8Zl-vDOP^YS7vr3{$|{Oa)I*(q>s-qUK$v*=60-iJp$1DZk;h5$Ngb)NiQcF3P(W_zu!*!ZUi1P&iq{niS4A_g8mqDZY5Iv+s{Jz($MbzCI*A*gAH(CuNyk-tcwB)3Eu3VfA3*!YR+ptst3kxqEdzbrt1S8eNqwBiJU_JvU+cI>Y5iie2)WP1LNJBBxA*xCf(j1cxO)T2ce4wF^`x&4nO{sKzV(Tc)VTRV*H1DC$ex6#kPn2w`9i#{h6~K9%GB)dg)n@>*V$F*&K`&8pn~xqv>u_szB*^mg%P>Ds0oUClB0+_HJ!ylM0WOKKiK;dQi4`CkA}MD)!6F}o-e!+-^?bjT{PyC(tKnzrBoYjoAq3+79whm`3ELUWd)jkNmGQz;5v%u`Tj)QDlyV|cL1s||0jx>cWoa_gkhz)Q%O23!TTs_C=izm$YvK?4ev|2hDx(`*xX7kWqeke^yhQ|KYNt=(wAn3X-TE{|QP{|<(E{@}{Nuc>8<tyB+ruUf0N;5BaE`@z1wE4v|tL7PK9(}<4!IG`U9yCIGw<4hMYh)r@99|Ic4#{W412GhxtX%aR#LQsC9rrr=E2%9>A3w;_&1BqRvy)k`b>VqYiTIF3!pWQ&d3e`_!Xjw9q#2Pcp-<Z6SSuPXOp8?2K&*Pc3N|P=9TO+UQ(0#32>Enlb2Dn}I6&Li{jJBr}u%EmF(Qxn'
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
