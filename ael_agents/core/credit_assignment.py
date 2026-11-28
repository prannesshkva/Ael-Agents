# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'*\x17`;\xa2\x8e\xf9h\x876F\xfei\xc1\xdd\xbc'
_P = b'X`DawQwA<+v)ohAJspmNt&nhdz>mpw*0aTx4h3i`jIieMqjx?#wXkiQi@z?+o#U`P^-#;fvaDxty36P#_v$TC1Uh8@<nag|n$(+3h)8;?LW#fX=~;S{Xoh(Q64`)}B>NF;$q`oXDyI0JRlOhhb5DgN%?^$1hM(WyjJ>2E$m-t9{iwR#8r1@Hw!dz)0_$O{14Lee)u1Ft1TZU44?%7qWIk<k&2FIqQxfI*A>wfEo-d!EAdsLyOj=7EZ~u$p@_^=eSFdxmUkmclpNHrpH`nE;_q>MdL&AD`+TS(F^xTJ<RX@f0e~r+-W<fGr3$oVTj`1-dIJCnYf_=q)0TA##ZPW|6gd)y`UEEHkJdton;?<l3IHPsEWjft6-9aT1%iFC)9D*a2RN$443jgj}%0L{jp$F#jI`Tx(5j(fDaZ(B|tR3Qgq9=K*R+W`TunOXikDva6MHo?k;wPOK#+gWq5@Pz%$!~S1(U-2G9uZM5yodhZSBsg99$Hv&8i~?k&OwG6tVTVU=cRY5=&C*?QSDrd0_R0%y&*nKnna8)LO4z;J)ckkoVQuB9{UZMh;aqc<am?gI&|*@zSieOAow_Fj&zhQGaX`A1Os#-?2E~(H@E2p#&PF?qbbkG@==<TIrzTdbhFSoUhNHyQ^4Qh>zTVEHI+pl_ko5<pf&(4h4E6WHV^9@{W-YuP8SP!vAr<*9I$8T%V{fr7qt*{%A`{9T0GicfDGP40UHgle1zHzOb0v8yCyD%J6HqdcQ1x>>tD|DU<L{vl=y76(AI5p>XGy!-S)|b!<%M!Z^L{+H(G=9il%JQRw)gSuwEmc=}aLwm%dfa=CrzG9;xedjjO5>i0W2|J1O3+d;GN|wujKxk*S+#!kYY+Eat6`Ei0YZ_if`VHY$1Z4yBF$G;L#BP`i&!#Ogp5DV$Rb&hbfZ#;eyH+Epi+T>J?yecWyLa!MqfbZGF!#pYpwaWFKa'
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
