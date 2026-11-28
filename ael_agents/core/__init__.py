# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\x80|,^/\xf4\xc2]\x8d\xa7\x01\x9f\xe1\x10\xd4\xe8'
_P = b'R;3xMT56sJn8Q!H-Y0p1>s8{y)_3^lKuN-Z%Cyl*QK9>`sUQqYnV=g!5<)X~DugJg@aohGm$kL7z6fwJnEo!nk!S!-8nKVQzkP%>^k|8g?4G_w!JqaSiA^PG^_ZGOEF-^W0;>t!YOM9qq;>bW=Vm#BO2NMQHa>1(PeDpWTC(1@8#?N%6dRdd9gCf`)e3UK&4)!uZ>3E8pN6l0N^7w;$fZb+^xjN3DE=Ck!?3w%9EQ=_9_s1phd=x(6(XMn(PJi2t$X^WaEju8ZWn@zE=oc1g!Eb@=4O}Scu>!wnJ^-V>b5|&h3<sZD;{hpc9?eOtV-TF?S)}K&&;RqJ7D=7A%n1fFB3DB$67pf@(?FepNvG*7xEOSk~?Y(<)+iR1@l;f<~zx_!qmlnk;0)e7^4AJKK<2cy`-hN4t{PJx31;~AKrQ9+NjE=vvY%2JY^txoV5w3GPr9(q!{(saXj1nIUKVv5}7?h@xu9LkGmRi2HUllkS9UwPqav>km?vu)>eQ5PBMl?Np?gwxj7g`M#~-mu|)#qBl8KQBboM@-+vNp<gjcj#w!|0hXm@m<x_M*BDsWnhQKAf{zvCYQKsXPH4XtXhCQ;7R1{1$>~f`<;ti=qj<zt$V>|n5M;SE=XmkB<Q9b?6GY9S1C%vy#-Arv~+Tu?a$HG#?GH94h{)$8Q$IPPkk+7%zC<JEucwEc4F|EX`=~)C9qJWW1P9_F01Ib%Z79LMNLFgfFbpKPWLJr)}R)5Xeh<GTG5d5||`-FR$pO`1CDRm#tcYbL(*s~j)gf6gkmjIqvprh_sQ|ER%X|%fDl>lOh9xM+-6j8T{^pq3~E%`8@%=lGOVfZUYGpk<$=kcFqfed&{YJ8eca_-7sgEL<a_UVN<#FmcKy#vMoEvfL`Y%5_bZ~}x?1$OV2l@_;QkS=@G<xkFXr_$5|369!uU5^n*Sd=JKTL;;Xm(+Pz&0GSB#;<v75u_W}2G?m$6J0SVp>bNjS!e}JMQ{qG^lpYf$#%&j%XWaod!uEV@R?4LM;_3XzpLx$^CX+5rNXbBuVA#z&8_n3Gyh(5jQ+ap*FZLTmSd%y^@silY@foC^tH)zc+}Ji9;zglWwdZ^8?1&q5T|CeUIPu4=K54fv1k__e)>0z+&V@Rk(0T_Xd#i|FAlfPL{%rR5ZI9@D*d)HD-_%eHn;z|xQHO1c0K@lqEGIio)viq80Z<#RxcC`E-8HP@>{E|=P7Ptk?qNL-yf%9{i%R`3{F4liMKn=7%hM8IdN@=ChM3btDmmUdy8*yszZX%(@%)f6SZTRI9n&7s&Wj4Us(RjA!b!46IlSTm6eiB3h$Ma+4xru&T117SHBk9^$bwLo#>_xlnOf$2opx@vlc4iB@e<zNQDasib8S!2bZGZD*SK%qDi0@wc~_p<I!UxK8se{lFK@J*0!-ZS*UAFo{@!;A1Sgm_S<g#uJ@h<@R;7C|N1H4{4#_v^If=Qhew*?O(P%BZju;1)B3F%wnE4N2>ly~J6@|S1aLzy!o4OxJ5!wYv5wmDnR$;2LPN~(#*uVw`I_5s+KuBaY3p|l&#pO#j%yLhdoU-L(G(v6D-LyUA}<~wuj(t*c|1Rr(fKsEcL*muPk}`4ioldGsHVFmQ!X#O(KG*t27w!E@dGlwashkwi0{?=XeME?;}Qui!%Pu!l8CY~kJtwMqUqSdpX31A31vOUT-)|zN?B8)<}u=fkVlh|a~(qP%8ZEXYevm)NMQ|Ef)Rl;uN?~dCizk->K>w(-wI(4a`Cn>JI4u)<IQO_(%=F7h4mF7(!yvM2Rn~2wJpAT`XK5^S`qdFzf0*6?Lzdp6_Dx7#u!DO0#E'
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
