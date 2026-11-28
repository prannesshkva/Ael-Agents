# AEL-AGENTS ARMORED BINARY MODULE // TAMPER-PROTECTED
_S = b'\xda\x9b\x8d\xa7M>\x8d1\xe1,\xeb$\xb3\xaa\xea\x89'
_P = b'ft1md9O#LT!l-(Ic-G=Sx{#<ADf3hcKSb3u^+^)6Qv5kJ#N0KbedoZ{&tMGp**)YATe*h`nZu96k3WhQr}GOFhZFm0J7XD+om3nX<qH2+gn!wYXlIi(VjsEQhp|qkh&BMUY~xxsv$)QmYm$(!Ft{^{KUg@m#nK;WLCMtQ7xz&iH+;a-dM{ZL2U8GS&#>ibHdcY4I9^<VCp+oN39DooP-;Rp`^{P~J)V+%qh>IGp6%v1^mdvtg}bUjr?%cXVfdZKhKS|PrE7Pc2xjKBoAQMqOT$z#Os2%Xa91z@&`$u0L%=*qApkk_;Oe%I2|zG=Vu&aYMK?ey03p@8f(pQ_irgw6LAzuCBzFw{!~7>_JXeiR6A8&`N{ZO;lR1Ua1i&3oQMIEvHv}Uz|6zIbw14PurzT_&vpTLXopeanizN#pDNkpZPQQ(WV3EMNt*4H<{ZnFH#Q)*Zp=h(G-G|tCz^!h4mayV?2kvVWs#u{1Kp|<Qu)JRjA!!$Sn?e^Lc3_caB%n--GDQq}sdO$RMX&3;CPl8d$79Z=DFF`YEE>M$!{<YN1OmI<zz1wSo(i<16$ixysgGm^B7Si~yU!*uk~5vwf~JhxX#pPS!g*9aXfNy7sG@JXAN*4}ui3ZobSfmYZ^YZb5m$K-gdU=o4#3wVbHZqlurC)t0>Bu<+SK<*T%@tCooeHkuW4HWzejSe>*_fZ0`{bMW&7Teo9Pa`p-lRDL^Mv4f)9*?HBXh}Q4A@_9A`dAU7slHf)M214xXoy0wW<wed<~z-t0z{wud9%odxS~Qe=+Z@GTVz5mL%vWqx{0))GFr=cRhM2S&})$V`Aj&1~q%`S>!S6BZo@?df>|Tfp@~lLX0LgD4u15FdVetZz;+dRl)f>C-%W62sqUg2xn~1Y=i2osRF9j{}FDCftXl!}s62W{FZEH&J1#dL2%*p)*6yICvGwJq30?@$o4W65O_0s6ts_ibJ%0Ci<1$O|TiYGUdBJ3WH8?C7t~+;SSLnbwP}w;GySX25xq5D=O7SbzQL`#eS)tL6|T(lP_cOsGqU!#RUL%MmAaf?Dt}j>vYL|Af##@pj2?zL}sNfvf*1Zdkj_'
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
