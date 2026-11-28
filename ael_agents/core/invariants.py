# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'3\x81nD\xfax\xba\x94./1\xba\x95\xdcr\xcb'
_P = b'G%Qke4Bof|*C-pyMK4WcaV(yCkALd9%X%slGKEQl5`eD}zU_H=4H&$v?LYa^uwfwQ@fuZs1e$^A8^Rz7Dy22SSZo4u4z<zkIvy6-Go)t~b4`FWRx~ixWu#R%+8qDaP>6c9p&474Gp$DII4*K2!W9v8hxJbteOD1OQ(}0zqFW)Ntt=KOO2|(tRt~O*7_>TWXqp2=oQhNel7|_I4YXaJ4HZW#SaksLS&lbcvgn!L*Qo!&%-Vau(NgdjQs641B^Km=JR_O=DQ?>qlqg?Gm~MlhMCcNqXgmZRRYg5!dzggNLH#Qz66Mf_vsW8eLWR9G<tI9uf`qGvk0H3UR!=Q1VHcTdH45bLJUUYak#3vp8hM|Y`Gn|R6uw7Nj4ulM%d#0xNhZK60E#`jM}nHmG=Gx?_`fHBIObIiy$5ufiD1T(;?b(ed%{m-DkIkmVu#KnohBAbNw%$&MSWd9EOy@b7RW(0Emyw<%Ob>Dne0oY-PMoVx~Z1x;(VkZ1MuQ=MOtnT;Qf#SaJg4i=m@#E^-6(uH#1fPbuztnu8?O0Kun-bx$Q2XnYtHXoBZjlAKE-;F`PjE#b%T%FnyenlxW5>w$^tPEWbA@W_j(rK%p;cbTf$l8*owo<yZ`~H$3QuJVQ`R!jfh^H3YvPk}*cd&yjH}51%hOB`g3P+64&xzDsv<{zt9v0y&(YX5?5u&T<xVZ^U_s$WEe6848*9H%9<PQA~S5Ay0!u;LGp<w538`s~?dm?IgnWhI^m`d8ZYly5<~K78@~9+*SF9pYF}VHF)(^c!};35=<ii8W49bkiKYW!p_pwn9Nrc!9{lC%DY-kCg;v)Y<w<G|EZAY6fg@87wnrz<?EG2q&D2cfg7+OufCwEp|g=Ryh`eElY>h8fbFw|XUExYhAl(Xk%fYY5Y&zphv0qY;-^v+y<qep@<oQXstQcD?gI#x^50iO<Ar>)HOI<=1#MJyP|~o)jNTJTAHznyb;Q>rkrvaskl)`hbz%srcEs>-4V%1l1lqY*f4)oZ*8SW_6JDC&8{b8C3S-&*rz$)JE$u)}Al$C}dG@R!Dv=@0YS%0{d2Bxn5(GYWwb=$gafYUd^w^b+@aB&0X>wPEUOeksj*RV_?1Cc1*6`&fnWqW^jG9t&)PgJ#b7goPw>v;<SIl4l;&u~do{wfzh2eyw*1H5Fv7Y=kN)|jLi4ButS2yc6G7BOG$9fz)?6s_wg{8d7'
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
