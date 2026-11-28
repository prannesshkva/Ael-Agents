# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\xef\xef\xeb\x8b\xd0\x08$\x82\x9d\xece"\xe5\x81\xa8\x1d'
_P = b'FK%)H1JsfN7yc8V*L>pk^TAW(xHj2RA~1!VbKm(o!pBW?Ar;;}-(lvm$5G25GV#a@NLm+{4cy7DJTcY!3j|&SIf6fy+H&Wexp|*DeukL-ssl8dabNB@BoiZ`jO<3z$$H;5l@9;Nhe9|La`BZDcEI5j>B!%HVoe_1%<1|<A*7dJCDWn<)SNMGMYw)2+-!z7m;3rf+O&Aw9Oq}}a6@EJG*)?TG%{wP0bq4e@ONVu-0UM7INS94GUmda8aw&f_~rRpx_zt#daoDW;f|Q|o?c|s(G?>Wsz0>=m<!d+tbJ_`n1Y!pKFH2KOCI?OHt+lB>gD)YIK7BUrl2r`p3z(M!{+?k1r}sIIB0qk*w_x*MrlT#G8dT#RAhe&eDjyyVn8>@tU}NQ(|&kA=1|w?$TTq}R;?VlC;$0D=8s#KUbOLn?(e>FI_Fa0e2nW|_VG#?dj7@s?ZPOPBeP<B$U>)5W^Pe<{Yt;W#J&UkkZRKk$O;BySZS}Dy(``@8^J8d_|}sI8yoX1&M|$|$=#ybm~4JJU=Rni0GgZUB(Tj5z)|{?$AJ@n6A3J6)CDAOS>+!M@or6NHDY!%)bd^}aJl9Y5<K5XvMG?Dk&o^dDG2CjlBo`j<~=7rnLswU9BCe>GcEEi?|BxS-fh!wRvo`1VX0jbMVupIBTu9K&?wT5IjRhSCss{${Q>9;0tNAlZL9hNwZ%TJK@>AAs&>E#%qf!qo}K7H$Qz_s8P?Ytnvn)t`EN$%{O{?cflb=;yA2!|b7OY}qYD-_=blXe(&D(;c&yx7i<PVHh#9pW8;|$#<jyM;R;78l90411;*)Ul|FO)@I`7vOfT2XGLI}JJLF45u_cme>Xv|cgS@?URh%datM0<}<85^PqA69%1=?Uqz%QzOfz<!KSEUu|9mwBx(=jq}4vUgGm`1@X>o+qw(_<sW~={M-%0(It7?1<fGv-DqEP6c#Wwk2QqKhh`Feh%-2575~8l&_3_OblQphdkP0kO{DBIXXm&5gblM^^I6hSx5;F@bf;XwDD&Zjdjw)Nl@gdx3`vmKwU;8H(1&{2a5Co!EaYQWG)Q'
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
