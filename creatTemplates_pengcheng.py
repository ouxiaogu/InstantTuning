import re
import numpy as np
import pandas as pd
import string

'''
This program generate a lua file defines N resist templates. each template includes all terms,and search a fixed point in the sigma range of each term.
The resist lua should copy and paste the following text.

AppInit=function()
    local attrtypes=TERM.get_component_attrtypes()
    NL=attrtypes[1]
    CF=attrtypes[2]
    SM=attrtypes[3]
end
TermFunc=function(algo)
	TERM.add_term(algo, "c_ap0*Ap(sigma_ap0,qtrFiltSize_ap0=20,b0_ap0)")
	TERM.add_term(algo, "c_bp0*Bp(sigma_bp0,qtrFiltSize_bp0=20,b0_bp0)")
	TERM.add_term(algo, "c_ap1*Ap(sigma_ap1,qtrFiltSize_ap1=20,b0_ap1)")
	TERM.add_term(algo, "c_bp1*Bp(sigma_bp1,qtrFiltSize_bp1=20,b0_bp1)")
	TERM.add_term(algo, "c_ap2*Ap(sigma_ap2,qtrFiltSize_ap2=20,b0_ap2)")
	TERM.add_term(algo, "c_bp2*Bp(sigma_bp2,qtrFiltSize_bp2=20,b0_bp2)")
	TERM.add_term(algo, "c_ap3*Ap(sigma_ap3,qtrFiltSize_ap3=20,b0_ap3)")
	TERM.add_term(algo, "c_bp3*Bp(sigma_bp3,qtrFiltSize_bp3=20,b0_bp3)")
	TERM.add_term(algo, "c_ap4*Ap(sigma_ap4,qtrFiltSize_ap4=20,b0_ap4)")
	TERM.add_term(algo, "c_bp4*Bp(sigma_bp4,qtrFiltSize_bp4=20,b0_bp4)")
	TERM.add_term(algo, "c_ap5*Ap(sigma_ap5,qtrFiltSize_ap5=20,b0_ap5)")
	TERM.add_term(algo, "c_bp5*Bp(sigma_bp5,qtrFiltSize_bp5=20,b0_bp5)")
	TERM.add_term(algo, "c_ap6*Ap(sigma_ap6,qtrFiltSize_ap6=20,b0_ap6)")
	TERM.add_term(algo, "c_bp6*Bp(sigma_bp6,qtrFiltSize_bp6=20,b0_bp6)")
	TERM.add_term(algo, "c_ap7*Ap(sigma_ap7,qtrFiltSize_ap7=20,b0_ap7)")
	TERM.add_term(algo, "c_bp7*Bp(sigma_bp7,qtrFiltSize_bp7=20,b0_bp7)")
	TERM.add_term(algo, "c_ap8*Ap(sigma_ap8,qtrFiltSize_ap8=20,b0_ap8)")
	TERM.add_term(algo, "c_bp8*Bp(sigma_bp8,qtrFiltSize_bp8=20,b0_bp8)")
	TERM.add_term(algo, "c_ap9*Ap(sigma_ap9,qtrFiltSize_ap9=20,b0_ap9)")
	TERM.add_term(algo, "c_bp9*Bp(sigma_bp9,qtrFiltSize_bp9=20,b0_bp9)")
	TERM.add_term(algo, "c_ap10*Ap(sigma_ap10,qtrFiltSize_ap10=20,b0_ap10)")
	TERM.add_term(algo, "c_bp10*Bp(sigma_bp10,qtrFiltSize_bp10=20,b0_bp10)")
	TERM.add_term(algo, "c_ap11*Ap(sigma_ap11,qtrFiltSize_ap11=20,b0_ap11)")
	TERM.add_term(algo, "c_bp11*Bp(sigma_bp11,qtrFiltSize_bp11=20,b0_bp11)")
	TERM.add_term(algo, "c_ap12*Ap(sigma_ap12,qtrFiltSize_ap12=20,b0_ap12)")
	TERM.add_term(algo, "c_bp12*Bp(sigma_bp12,qtrFiltSize_bp12=20,b0_bp12)")

    TERM.add_term(algo, "cMM3*MM3(sigmaMM3,qtrFiltSizeMM3=24)")
    TERM.add_term(algo, "cMM4*MM4(sigmaMM4,qtrFiltSizeMM4=24)")
    TERM.add_term(algo, "cMGS*MGS(sigmaMGS,qtrFiltSizeMGS=24)")
    TERM.add_term(algo, "cMGS1*MGS1(sigmaMGS1,qtrFiltSizeMGS1=24)")
    TERM.add_term(algo, "cDE1*DE1(sigmaDE1,qtrFiltSizeDE1=24)")
    TERM.add_term(algo, "cCSO*CSO(sigma2DCSO=0.01,sigmaCSO,qtrFiltSize2DCSO=5,qtrFiltSizeCSO=25)")
    TERM.add_term(algo, "cCDO1*CDO1(sigma2DCDO1=0.01,sigmaCDO1,qtrFiltSize2DCDO1=5,qtrFiltSizeCDO1=25)")
    TERM.add_term(algo, "cCDI1*CDI1(sigma2DCDI1=0.01,sigmaCDI1,qtrFiltSize2DCDI1=5,qtrFiltSizeCDI1=25)")
end

--resist:cAG1=0,resist:cAp=0,resist:cBp=0,resist:cAm=0,resist:cBn=0,resist:cMG1=0,resist:cMG2=0,resist:cMGS1=0,resist:cMGS=0,resist:cCSO=0,resist:cCDO1=0,resist:theta=0

'''

template1='''template1 = {
    modeltype = "fem",
    role = "",
    templatename = "template1",
    useterms = {"cMav","cA","cAG1","cMG1","Slope"},
    sigmaAG1 = range1,
    sigmaMG1 = range1,

	b0_ap0 = brang0,
	b0_bp0 = brang0,
	sigma_ap0=range1,
	sigma_bp0=range1,
	b0_ap1 = brang1,
	b0_bp1 = brang1,
	sigma_ap1=range1,
	sigma_bp1=range1,
	b0_ap2 = brang2,
	b0_bp2 = brang2,
	sigma_ap2=range1,
	sigma_bp2=range1,
	b0_ap3 = brang3,
	b0_bp3 = brang3,
	sigma_ap3=range1,
	sigma_bp3=range1,
	b0_ap4 = brang4,
	b0_bp4 = brang4,
	sigma_ap4=range1,
	sigma_bp4=range1,
	b0_ap5 = brang5,
	b0_bp5 = brang5,
	sigma_ap5=range1,
	sigma_bp5=range1,
	b0_ap6 = brang6,
	b0_bp6 = brang6,
	sigma_ap6=range1,
	sigma_bp6=range1,
	b0_ap7 = brang7,
	b0_bp7 = brang7,
	sigma_ap7=range1,
	sigma_bp7=range1,
	b0_ap8 = brang8,
	b0_bp8 = brang8,
	sigma_ap8=range1,
	sigma_bp8=range1,
	b0_ap9 = brang9,
	b0_bp9 = brang9,
	sigma_ap9=range1,
	sigma_bp9=range1,

     lua= jobpath..luapath

}
'''

##########the sample number in sigma range
N=100
terms=['AG1','MG1','ap','bp']#,'MM3','MM4','MGS','MGS1','DE1','CSO','CDO1','CDI1']
########ranget records the sample information of sigma for each term, modify if needed
ranget=pd.DataFrame(columns=terms)
########default sigma sample range
defaultrange=np.linspace(30,300,N)
for t in ranget:
    ranget[t]=defaultrange
########set a special range for the terms that you want it be different from the default
ranget['AG1']=np.concatenate((np.linspace(0,20,N/3),np.linspace(50,300,N-(N/3))))
#ranget['MG1']=np.linspace(70,300,N)
#ranget['ap']=np.linspace(40,200,N)
#ranget['bp']=np.linspace(40,200,N)
#ranget['MM3']=np.linspace(40,200,N)
#ranget['MM4']=np.linspace(40,200,N)
#ranget['MGS']=np.linspace(30,250,N)
#ranget['MGS1']=np.linspace(30,250,N)
#ranget['DE1']=np.linspace(30,160,N)
#ranget['CSO']=np.linspace(40,200,N)
#ranget['CDO1']=np.linspace(40,200,N)
#ranget['CDI1']=np.linspace(40,200,N)


temp='''
jobpath = DM.get_jobpath()
luapath ="/h/data/dummydb/calibrate/job1/lua/resist.lua"

brang0=populate(0.1, 0.1, 0.2)
brang1=populate(0.3, 0.3, 0.2)
brang2=populate(0.5, 0.5, 0.2)
brang3=populate(0.7, 0.7, 0.2)
brang4=populate(0.9, 0.9, 0.2)
brang5=populate(1.1, 1.1, 0.2)
brang6=populate(1.3, 1.3, 0.2)
brang7=populate(1.5, 1.5, 0.2)
brang8=populate(1.7, 1.7, 0.2)
brang9=populate(1.9, 1.9, 0.2)

'''
templatenamelist=string.join(['template'+str(i+1) for i in range(N)],',')
temp += '--'+templatenamelist;temp+='\n'
for i in range(N):
    print i
    s=template1
    s=re.subn(r'template1','template'+str(i+1),s)[0]
    for t in ranget:
        s=re.subn(r'(?P<pre>sigma_?{}\d*\s*=\s*)(range1)'.format(t),r'\g<pre>'+'populate({},{},0.01)'.format(ranget.loc[i,t]/1000,ranget.loc[i,t]/1000),s)[0]
    temp += s


f=open(r'C:\Users\hshi\Documents\softwares\python\pandas\gaugesignalstudy\processResistTemplate\defineTemplate_finesigma_pengcheng.lua','w')
f.write(temp)
f.close()
