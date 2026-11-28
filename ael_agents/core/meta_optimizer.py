# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\xda\x13\xc7C\xd3\xd2\x81\xb6\xaa\xe3l\r\x82Y\x08\xa9'
_P = b'YW)$&@LS?#Ddq}I8*sVzBW`8OICf*W%MEtZ_n1>@VLD5T1R{FI$&5@=9DrJ``M~?T6e_2)X^A=SaIwJ?;T1CP%AArtN6sZ<x5H5rOpBX%f?k9yN6Oj5sa=x;xd#k4m*=CG7lK<%BY@e?2r6iP6M*7|cmAhtos)x6y)QAL7;g4iCoU96+0`&_-l^PZEymz7e<OOtrz}c1O-UuGb_~K?Ghi~-&K;-d;=TMN15&<s3-{eHQ_WSy|MI+SD|FF*4(Am!=_!5GpFuo~0CN5!XKZQz)&3er54P+MjQO?t#V3`c_<%LSO0Mj6$7z&_SRi5ik3KClj^;JHTK=jvV@_by#71+B-=G(8wVUg!NZ%`Wdp9NkXbDjs6PtN@12;q4d9Aqv<*^G6^Iw{(8Bu~M19IEQo7my(NN4&b3D*T!6gA-}MV82sd*qn@w$YgFQ?2Jwj>x&#<UY(GpC5v>RkgO-+$JiHj38=Jd86~?`XhKHia<;s-X2M_hI6#_$FGs+opW&dHxp-=Boz13fs-1;-Qgg#RFSSOwJ*;_3qe4cHUTLuhZ^wd*+}e=Fj2$1#gGP`-=_XUMaOHn*XAkSN^gA@@poui9(yd{>I6l>$Y9(O)ZvinY$O7H_vV;x!0-zs<3Dq}<mw?e6D3*I%&tsF(7oVeW=BH8<<wM0IEP$P35r;t>~F$1O2{b$+cs5Ey%=?Crg2rYCZ1BHSB1Y(L!^N#ZB#)5m*;5mgM_c6<4^y%HzLa1attX{C^=QN5zw#{;9zmMsRwXrTFGt+<)sH8V`K}ZmvL#9l>!}|;<4T~fq1sg=IDrhNu9lDZ4~-Wfwt7`1OamsdUQP4|C}UuzyDZJ?=sD607uij5{YjP*?G$6|Kq?%hX'
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
