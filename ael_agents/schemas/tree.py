# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\xd6\xd9\xd1/\xed\xa8`\x8a\xbfFc\xb2\x0fO\xbf\x80'
_P = b'Y}GVG{0!(*Z(Y-~rN%%P2UmgaS0)rK(q*Z(`08Gb1U8KXq+`o}O=2e_-KV9kDXI@(;^{kFc*L!F554!v2wx`t%`R9J2J|gyu1t|DlJsw{8;Lxl(YOq%886oLn<Yj30)d;A=p;v<Zsg^ASXf%iZI%}x{F8fuCK_FFM-jFiW+Tf}s7?bmPB(ONue+JZ6VexlhAx3G59p`v)R0S0K%_BQw?}=MXIiaIRg)}wy2=LT)4Sp7q!9fnuIc2>ER%yzFnub#+hA~p>3sMXB`?8ZT!n319B&A|Y?856rMi_5T)9vbY<~EAV_$sys%eo*%!c_k1yP@NyHhaffpyHID;eFw2Asf^{LG1Zk^=Ip5-H3<_5T8!_B&0Jw5C>2oMtpp><%m%2L<()K6+YxD)RkgY8U)4bE_vDba&hfJ6eBG9`{ipou*G@32p?RqgZxkkrvsLEpxm6!zbnuaE)hvJS~i)urGI6zrqqL*Kbudzi{LN8O>9aCP(3yONb)S(@<Zz`@K%8=4tx&Bag07|H1!wyy4$C!+a;67&Q^JS5Mx=!rA-4zoe`JD^h<3zd8cK``K^Y0si>iD)w1DGlqV9K9hIYQsu1VaB3Fl=hSlyZPs1%s4gVdDvO$(-wCCF)YAhrxCDGhF^l!3Xd_7-k6eY;$t(~_RIfO9;k#x6#|RE%T)Xf$TSiM;RS#*-5+|`>-$L4$gvkFK*7Mv;r74K$Z9*D@Oq%_{hJ)Q_z}(^y$ht=YtW{869edUAQSNioMc+A>%5aFn<Bm-^Lq$7-<qBgPbHcY8md|2FG@z8~p5U1zUhQ~WOB&9JNBFT(PA7sHVHWmIVC}_#3MX26Wn2`u2lP`L<N_@>34mXWYW00wY6iA2siS`2rqYm6|Ir6ms#{(A%FxZbKYA_%fc@-_of$i7Z{1UH_M1T`s(lIq^RU6Z<t`k^e+_7e_cp4at$mY%sTkED)E%QA!Q``}hAt~@s6s&;$99$HpkL1j$Um~imMrcIg0@fPph9XGAHLEB4Se^MhH*`L9@V15Ww2qg;B3vM$-W}<0nyALKVaaeHiE?Cw!up-fPfa)m>FUEb=U4|`jxs`;sgZ^dy#z^8ldSPDb2(8;hrq2!P}(=d{z0+M3_xK2^xo!2nU%-p~<X*9v%wv`J<>~cD@OiWATQGx61sLo<{v)jyMT=uaj)+S(c=ZUHQQ}lM5=Y-A+Q&ieadc?S8u|QN`Ud$?xSbvQ8UYQ=WXS%tn~KAxsIM6-T0jyyd%)4W{6(rTnSBHdSWzg<#_eICzX@`kYEg$CdXek<wjPTe=8NxsMolmpLCsmke6t0{>@RnyHF*Zfu*^&FbxI@iTYv7QOF;BHWzLwfs|8Lbqd?UNxVuO~jL5J@P-ht}YxR(NKXH6r0n5V-(sZ#~O>W$3W}V|MT`BC^a-IpEeIReXJv(;0^'
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
