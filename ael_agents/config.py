# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\xf4\x1f\xfd\xe8\x9c\xb4\xf5\x04\xbe\x98\xc7v\xa0\xe5\x87\x01'
_P = b'i%!AFs4>cC_K!cU_~@;wMp}3I$^F7(<{R*=V}$%g>Z;m<k*-SU4B5*5wF2X}BLqK@iTd=N8BY8<;w>`s&T-fyr?f|CZri13cDkL5<`6M`?Ck%=JsvufmD=;oRTy7gzV&gI6R$aG3K(zW#oDWl5OKt+Yx&tiqZH?auB#3WhjAm=pd460d`Ugya@IumvpB`9n(0M(P^wx2_xhbDJAnnWcs!fsI~%2B!ApUfm-@(Ag{UO=2T1)^aZTCYh<P9;HIMnWe}xq!Ecc8vYA-fHx2^j=sC1Yj|Kj$JX^R-^Mq7N7$yA37nV4><EtX-V{6sz29|W$~!cOMUFJ(6Ukw(n~^@;3qxjBF7#LJ*Ws7<Z=X^xo*hB%<nOY+Z4(;B|+8;@d13_LnUzGYlGo}DiD<sVId8kLEv@`18u(|e{u`%#*5iN8Y2i>)(MA-gWy_1iJ{b}f_|0mhJzP`NOl7%PDPXD$6`ef%U+XShxqs=A>wxQC*Tz9WWdP3)ryp#IJTO0A(zg!Y#^DY}f2U~M1oQc>ldBEuz!DR$X)sT_{|W=O-t78Q4qY>&}OX@qb}Z&y38l#gT@yQsNgo41_as_j5DYDOY_2D|`%>>Pj^MaY#dQfPWE@gIS59&!Ny4%mTooVaV{W-U_H|DR(TZt1DzeH+w#oef_N3y|3mb0hjf74UPU4hn;lAeIr^>?q|?0-Q2LBtt%pa)D4%VOu!>Y=*6$5Zt2NfR!sp!S|mxW>d5syv{oSh-Epv9p5L_-4vdztTdrFuptz27ayh9IfL+quCAdx^)KiK+ZtL3CvSs?<DZDdyJdEI36%bgr>1<K3C*?rOM?DES@~@Hhw3VG$7S`ITVKe>Qg4`gMc4e^g4g1;qpRdE;Q|}*aRqMN<FR-=96-|L!TDGU%VTf%Q{W9-9yF$T!Rncv0!57pSmYkl{f{XFmarFZWHGI?E3~<XhUjE~P)&YgI!cRd7chL}bR$4yHdb>JFAEPg%-%{QC;9(_Qa=-mCOv`5+Occv%))3ykK-eTu?+S*lIsB|AMyqaEFrjAeSF6FgvoB|Q>e}=VNbyTBbi0bR3QVQeX=LrzvH+c{$B1JZgl744KiFrQ&?9uYvZP`?eLDX%05iZh`zS58$zly#o)yqB;zhWfX~u5H!FSvJfw}%6OdQcap<U#g<Lu<0j$UlwF1JH+KM(CA*X^Y;!m9thCm2<x6LwKZa0=#SG+XPv1E*StWdo%K@<-`gK~%CDh0X;LBmy!1Zkt@{DT4Ffy3^nP<YOn<TyJ*otBF+p+K4Z!g|s)yO7L5?AEZiqa|tCW#bm(vWz(-C(VMu%flIS!(T!|AOacrJgnVYLYIowvgJ5QZKeuQ<+q|G_CNP|Q!MJWQXHvM^iwCDJ)(_gnf}s9w8@jiUtrVmpi@^MNj&!WAy0_>sq83oAxjhfFVC2%hcP|{dzz6Pj-j2YNQr2|A}t=*pogFGzg^afLYVBNc%3`1E*<%niOECf&DZ}EM7FgJ1tOKgaGXxblJiU;m;hV)=*C6s4u8_E{!DOSB~?AGu<Ln4+eKFUklu<9<jU#Q@<<1h1t>k>5<%%2sDdjKX}wd$1)Z7ukLD@FcOMahCMg6j?P4XTDr{&B373&{c{eOj$}(=*<h?`UioUs77o@lGt-mg6Q|0|C?9EYFqbs)LOHLZy1Fbc(fhu_4zYR_1'
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
