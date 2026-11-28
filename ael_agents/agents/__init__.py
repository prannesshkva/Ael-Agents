# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'C\xc6x\xad\x15\xa2\xd1\xdb\xfc=u\xae\xf9\x98\x1e\xf9'
_P = b'-5GmQF`>^i1D$p!|8$VanUlg<zhmvC6lYMc_5kpPew03fFIk_arH}gR@#o`iTK5?5iurO16TTB9mCs$>$q)6*bc3x$;{`%Vg?BNxJ{xe#dDnIF<^|OIJBonsc8TKBd3_ay2R7DO;2t-pYjF>{ZU|wjP;*g-$HYjzc<Cpet2d)JlyZIzn#n*1qP_coL}Cg6VC8&#Bcv26U~$3yl_pW>07Tkfa`n1%fYt}QYX9N;Zva2miv1-u?0c2$6Sc;T62wFx;CA~9<dx+K+a`Z1q@Y(5<URF%V?j3@H#;cB4;^7w0jDSJAivXV(;l_ZOvu9Q!SaM`Jp%4y0TjsOB1&dFINIaUO0W9JGTEKv@``~%&sIRVb?$Zn`NYyx07#a^JIziINJ^wjK$6voz8rg;+>Z7ql#sT&f%=7TNq031WSRkL_GY>xiu&p%Jd>Os$+b^g!t;yI6DL~#h223RHsyswDE%C(3m~3eUw8>(VGYVe*YXa5V6BWtx%y2mY)ik>gL5hP_lG7b;_6wXhcLq0Qhwkt3F%iat~IC|F*8H0sj2a0*f2UO`9*3QbPVBkHqxV6t>E;k9<x*'
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
