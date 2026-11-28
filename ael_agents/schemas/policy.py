# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'7\xec)\x11\xf1!\xd2O\xa7\x15\xdb\xe4\x14\x90\x91\xde'
_P = b'qL>REF#vJsm6*rc=RjxTd>SNY4=JiP)wUK*$uyurMW%2xe_l9K(P?x5o(V>?3>e4}(>_f_NsU3kUr5$Ex!(u+0I9Qielzn_V9^&@s(Ci(rX!!nx<QrO{e<FH(OXqAZs#CK%om|!l}2Q**C--(Bh5ZNSJ@`4ko;LwDMFpM=?&mTFN^>x^dQ}3nFxj3PgQ6!JjqYB<KOv}Fa+iT{?PZn$BW#gGx(&?;;F{rX?Jqyo$&%I!$Ps(a=iO`RWsq)ynS1O+b{_e%7h*ayE8hX@;8mt-ljY^@OH#6t$kkrT0qs>XVCWiMvMl4=4uUydy#W7GR#W|+W!D@LpTtcr|i?qqQNe&l80n#Ig}o?eg=N$U-*ZmePhW+&U;Ex+xPuDT#9Z89#c-RO@kzWb2jg<Ged1i=yV?Kfp8~_fAnz7Lul%Bh3@Dj5?Mm&Yr20%7;MnZpH71@y%Go04vJN<m)0ZaW)VqjYAJ?@VCt==>eZtNG}AI%qTB(cVsE@eknjwB{OHD@BiVVr(8F`aaE2#@=q^QSl1|NBbFK!-!mnka-}myR8+hc=j*k+H0h^6jKJBk1q;`@KSfxhkxYU@Xx|iRo(Ki+KF>2TDMH2z$YY}O7YAUZYdqzszl4v;v=I1!^LWzY`Hp+AR-fXT5IgmJbu?}*0Vcm1QTvT_X6LrQDd&l3Q$Y|VY+F8?1E#yS4E_#S@xhm|q2sPUdLb?4HyC?(!kd`5Q_MR0t8+M7TKbFj*C<~rz{Tx~-LtU2;5-!K5py$-0$d8xHSmI7U!Qy6BtF!5jg#VF0Wh=vR&8qw@4SsI<IzfUhfcb<%s|BkO15Km{U|OEJ?1l=xUP99+*P4C<9NO?_)q@gQ&$~eQUT$eS8g6fEOKDW4mgE<a-TD@2aEc7bW>{@Swlj0{&Zi1~Ug=rt=HZI({Oz5nNZBJQP0)<&?`;s|#{diVCnXNDJMA9(e6ghv;z*%=>i)x-bulK!ZP@t0@|}n3P>u6G!y0ecx;J3+hR5jlW-K}i!cwot)+*_!bvD_yBE<4dw_i3B*@NdA;r~GN`-LK?WSH?u;WdXfFFehg)x=OW*+H-IC<bpVQ1$SEt5BabSCiBhJUWynm8XFOT)fPbTn!;yuei~HF7z&!a1{qL0YSef%^3Kp1j!Y9=$*lx)ZRLfm(rCn%UHiC7GV9=%MObS>id@^%rU+=Q8)dk=p$2UKs7h()q}!B2DN`-jj{F!6NV42%FyjtKLpF$Mub95%2)~-)+*g75?iGW|40mQK0@ZIdA>}|NdDP?CU4=!xN$)#__6>#z^AdK>DUN08Og!1JIAW>^6;hh^-eFZv|TUUf4tjKY*1BKGIOMAX%2?VJ+VC;Gf_FKMYZcdL{j>{E}HRwV0{!TnR<(ln|*iA5=ZC2Z$io&d0DhD%mE(`BjBh@0*Ag1YoNG5L;G_g9&sK1mYMn*NfUj6iudIs?m4~}*Z&skF6(&Pop3?n@~bs1(}OYHUPr9=Aoj=>Lvaa)wgXd5OydQLnOkjT{O-GILIGg-t&m`-xMwTpRNEmmijW4Cf9NMK;4beJ+VfdyVWbWg%$${L1_WEV#9b(oea&P*6tkSc;Ok6SQl*h%FNiLoR=jsLxJe@f34u>>&|jfLvByE=Yom2JZ^Is|h?&f(QViB;YOmBfgvw1h_DE$8AKcZ)_Y;?ab@AzDe7+e&BVc0kd<IHcKQ?p2uljVerww<r)G3^mjl{Y_3o&RId59^hAV`g_**3`EtXVh5H{^6W>IjAi`kf@2xh2)pVLL})`;~XJ@PoyK1)RMrCb>iDWuv8W(5{Q^pXXtrS5{YSA&Oj87<UGW|3Iwjs}NH(R%{oG'
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
