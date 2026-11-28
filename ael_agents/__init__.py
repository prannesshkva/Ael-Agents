# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\xc4f\xb3X]\xea\xa6A)\xf5\x15~w\x14H\xe4'
_P = b'r+^bTh37mEbOrBuC&H&~8{{-=F+VC@3-#~7*faDyY?&G)xYiarbTHNlE~c|Z!i*|cFTJ^y=SZ#uD+Bq?SjMuN&n)pvV1qn*vn7VP&B8+$86Q9A6(?zSm<9)2U?Vl>e$ASLUN39yl$zI9S#xS*F$qc)W%AF3O)Huj!N&EX>1DcK!P9YI{hG~(V567<8?m$;K<jL>EWa8nUP22q&sp01f=G(-sC|pLoBwQtAgzqbx>!d2Nij*iqlfv|Sw9vYM33z_d|QX=MRDzvvcg^!np;>Olojhjitu|iIQ+ZSb<LX@!6k-mD+NORKMeKt^Sf!jH2dmWWfobZwg$swknhf)LJz{w&SUT1Pw|WreN`uKfmrOHKSYL!G0Wb<uUA{`WV0Tz=G;Qguecb!uetEdzVxQzh>c3cy1h{?a*$FbDrt|R`Mga^C+op9KO2c8Iy<LuthUU8Jc@Ny!z%K%A&=8EvA_FUB!fIAA?8cwZfsxOoZVYbSU7Lv1oxHN3>|%eqmnIorYkyOa(^t?^dEaJ8fD@1&M{|mq%KL_&pFa%`ktD1aek8xKy3W30p&x&I;sBGr{M*L>+ZYD%?V?$Sy+<KZWgAfePt!BB<r1U?Bbnzu1Km#+tKd(^!TzM!}*5$&gS~@T614HRnh)%X9XdyO7I2)pUhkdPu4xvFUfQl*(e28QN`~FWpzDdID!P1j7dA#<IMHPhI-SdJEln%;n6hL%B(Ud{c%WJtC|fWSLKR#?b^rj`Y`NGZBAc0Ex;{E;h<M(h6fwTt}KL!#IKSFXqS{iDSvvmkU**!idy*e$qdh9G!&%B56*nT5b`PZaZ8V6+2t4O0xp;kh*>m1N>j0#+CkP|Z2HIMt}S-Tj*X)4iE=-V6Y|e`28vj(@cGLGx=J&@JRhX3K!1xq{0irVwZt+*p0Wi@>SuEv3ax<3joB-q*sYRQhb1Tfi>Z^DibK$r9?zq~jj)HOL9HJov&At<cDZvz@3tCv!%daKMvRsM$^bP@wmMelDJ*65<LfiFubwX22&!5NLq1jaxhTM3xb`WkKcMnjj(HDhGWX^18tx$clza+gE^dJF%%5gHs~3epJN`*u_H>FtSr)l{4m|-7Me4BNHaS5umxVA35oo8Sm-C23@#}4Q4g@2#Mv<U(*-8_l-Qs8gi<DZ0;@H9BUb{2z-3Q!NUf{wfs9zG4d?BW}U%NDRppoRu=_-|Y?i8f<7hcHZiyep=>jQ<fdQ;)L7a#_Q8{#Ax_BOwFjF5Yr2?s$?d`Em9iVx9Uc}Y#Z6e6{22f8}T%wqH}8yfkQ#(%##ZWQG8V$!)m%eC`kfs2wH-mpQ@pkCrL73tkxZ9pEPFQ;QFh{)MbFo764@c8o@PK8I?9gIi'
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
