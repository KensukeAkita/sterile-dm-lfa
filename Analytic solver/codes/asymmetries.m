(* ::Package:: *)

(* ::Input::Initialization:: *)
BeginPackage["asymmetries`"]

ThermoQuantitiesC::usage="ThermoQuantitiesC[T, \[Zeta]\[Nu]e, \[Zeta]\[Nu]\[Mu], \[Zeta]\[Nu]\[Tau], \[Zeta]B, \[Zeta]Q] returns {\[Rho],s,P,\[Rho]TotCorr,\[Rho]\[Nu]\[Alpha],\[Rho]l\[Alpha],\[CapitalDelta]N\[Nu]\[Alpha],\[CapitalDelta]N\[Nu]\[Mu],\[CapitalDelta]N\[Nu]\[Tau],\[CapitalDelta]N\[Nu]Tot,\[CapitalDelta]Nl\[Alpha],\[CapitalDelta]Nl\[Mu],\[CapitalDelta]Nl\[Tau],\[CapitalDelta]NlTot,\[CapitalDelta]Nu,\[CapitalDelta]Nd,\[CapitalDelta]Ns,\[CapitalDelta]Nc,\[CapitalDelta]N\[Pi],\[CapitalDelta]Qstrong,\[Chi]2B,\[Chi]2Q,\[Chi]11,Nl\[Alpha],Nl\[Mu],Nl\[Tau],N\[Nu]\[Alpha],N\[Nu]\[Mu],N\[Nu]\[Tau],\[Xi]\[Pi]c,\[CapitalDelta]N\[Pi]}.";
SolverAsymmetriesNoNB::usage="SolverAsymmetriesNoNB[Tmin,Tmax,Le,L\[Mu],L\[Tau]] defines the reduced neutrino and lepton asymmetries \[Mu]/T (\[Mu]behavior[x,Le,L\[Mu],L\[Tau]], where x (in brackets) is e,mu,tau,ve,vmu,vtau) given the initial lepton asymmetries Le, Lmu, Ltau, assuming that the impact of sterile neutrinos is negligible.";
ThermodynamicsAndAsymmetriesTotal::usage="ThermodynamicsAndAsymmetriesTotal[Tmin,Tmax,Le,L\[Mu],L\[Tau]] returns the tabulated data {Tvals,Hvals,gsvals,\[Rho]vals,pvals,svals,\[Mu]T\[Nu]es,\[Mu]T\[Nu]\[Mu]s,\[Mu]T\[Nu]\[Tau]s,\[Mu]Tes,\[Mu]T\[Mu]s,\[Mu]T\[Tau]s,\[Mu]TQs,\[CapitalDelta]LStrong,dtdTvals}, all in the units of GeV";
TabRes::usage=""
Begin["`Private`"]


(* ::Section:: *)
(*V2*)


(* ::Subsection:: *)
(*Constants and definitions*)


(* ::Input::Initialization:: *)
(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)(*SECTION 0\[Dash]GLOBAL CONSTANTS,HELPERS,DATA FILES*)(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)(*\:25ba 0.0 utility for in-lining into CompiledFunction*)
SetAttributes[ruleDown,HoldAll];
ruleDown[list_]:=Flatten[DownValues/@Unevaluated@list];
SetAttributes[ruleOwn,HoldAll];
ruleOwn[list_]:=Flatten[OwnValues/@Unevaluated@list];

(*quick sanity-check macro (optional)*)
checkIfCompiled[cf_CompiledFunction]:=Module[{out=ToString[CompiledFunctionTools`CompilePrint[cf],InputForm]},{!StringContainsQ[out,"MainEvaluate"],StringCases[out,"MainEvaluate"~~"["~~Shortest[__]~~"]"]}]

(*\:25ba 0.1 Standard-Model& QCD constants (all MeV-based)*)
\[Alpha]EM=7.2973525*^-3;
eEM=Sqrt[4 \[Pi] \[Alpha]EM];
GF=1.1663787*^-11;
mp=938.;
mn=939.
mW=8.0377*10^4;
sW=Sqrt[0.231];
mZ=mW /Sqrt[1-sW^2];
GNewton=6.70833*^-45;
mpl=1/Sqrt[GNewton];          (*Planck mass*)

GammaE=0.5772156649015329; (*Euler\[Dash]Mascheroni*)
LambdaMS=200.;               (*\[CapitalLambda]_MS\:203ebar in MeV*)
muref=2000.;              (*reference \[Mu] for running masses (MeV)*)

(*quark& lepton pole masses*)
me=0.5109989;mmu=105.66;mtau=1776.86;
muq=2.16;mdq=4.7;mcq=1273.;msq=93.5;

(*QCD colour factors\[Dash]-needed by CorAsyQCD*)
Nc=3;Nf=3;
dA=Nc^2-1;
TF=Nf/2.;
CA=Nc;
CF=(Nc^2-1)/(2 Nc);

(*cosmological/model parameters*)
LambdaQCD=150.;     (*switching point to hadron gas*)
Tcut=280.;     (*upper limit for lattice\[Dash]\[Chi] data*)
\[Eta]B=8.6*^-11; (*observed baryon asymmetry*)

(*\:25ba 1.0 momentum grid for lepton& quark integrals-------------------*)
n3=41;
yMin3=0.01;yMax3=20.;
dy3=N[(yMax3-yMin3)/(n3-1)];

simpsonWeights[nOdd_Integer?OddQ,\[CapitalDelta]_?NumericQ]:=Module[{w=ConstantArray[(2 \[CapitalDelta])/3,nOdd]},w[[1]]=w[[-1]]=\[CapitalDelta]/3;(*end points*)w[[2;; ;;2]]*=2;(*odd indices*)w];

coeSimps3=simpsonWeights[n3,dy3];

(*\:25ba 2.1 Simpson grid for the QCD integrals---------------------------------------*)
n4=21;
yMin4=0.01;
yMax4=80.;
dy4=N[(yMax4-yMin4)/(n4-1)];
coeSimps4=simpsonWeights[n4,dy4];     (*helper defined earlier*)

(*\:25ba 2.0 lattice-QCD susceptibility table-----------------------------*)
\[Chi]Table=N@Drop[Import[FileNameJoin[{NotebookDirectory[],"data","ChiTable.dat"}],"Table"],1];
tGrid=\[Chi]Table[[All,1]];
nChi=Length[tGrid];
\[Chi]2Bgrid=\[Chi]Table[[All,2]];
\[Chi]2Qgrid=\[Chi]Table[[All,3]];
\[Chi]11grid=\[Chi]Table[[All,4]];

s2B=Differences[\[Chi]2Bgrid]/Differences[tGrid];
s2Q=Differences[\[Chi]2Qgrid]/Differences[tGrid];
s11=Differences[\[Chi]11grid]/Differences[tGrid];
(*\:25ba 2.1 fast binary-search& linear interpolation (fully compiled)*)
findIndexC=Compile[{{x,_Real},{v,_Real,1},{n,_Integer}},Module[{lo=1,hi=n,mid},Which[x<=v[[1]],1,x>=v[[n]],n-1,True,While[hi-lo>1,mid=lo+Quotient[hi-lo,2];
If[x>=v[[mid]],lo=mid,hi=mid]];
lo]],CompilationTarget->"C",RuntimeOptions->"Speed"];

interp1C=With[{idx=findIndexC},Compile[{{x,_Real},{xv,_Real,1},{yv,_Real,1},{sv,_Real,1},{n,_Integer}},Module[{i=idx[x,xv,n]},If[i==n,yv[[i]],yv[[i]]+(x-xv[[i]]) sv[[i]]]],CompilationTarget->"C",RuntimeOptions->"Speed",CompilationOptions->{"InlineExternalDefinitions"->True,"InlineCompiledFunctions"->True}]];

(* \:25ba 3.0  extra masses that appear only in thermal fits --------------- *)
mpi0 = 134.98;   mpic = 139.58;
M1   = 500.;     M2   = 770.;     M3 = 1200.;     M4 = 2000.;
fFD[y_,\[Mu]_]=1./(Exp[y-\[Mu]]+1.);
fBE[y_,\[Mu]_]=1./(Exp[y-\[Mu]]-1.);



(* ::Subsection:: *)
(*QCD piece*)


(* ::Input::Initialization:: *)
(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)
(*PART 2\[Dash]QCD correction kernel+full g\[Rho]/gS thermodynamics*)(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)


(*\:25ba 2.2 compiled CorAsyQCD-------------------------------------------------------*)
CorAsyQCD=Hold@Compile[{{T,_Real},{\[Zeta]u,_Real},{\[Zeta]c,_Real},{\[Zeta]d,_Real},{\[Zeta]s,_Real}},Module[{mubar,gS2,mcor,muCor,mdCor,mcCor,msCor,xQList,\[Zeta]QList,\[Alpha]E2=0.,d\[Alpha]E2d\[Mu]=0.,\[Alpha]E7=0.,d\[Alpha]E7d\[Mu]=0.,i,j,y1,y2,fQ,fQbar,dfQd\[Mu],dfQbard\[Mu],F2,F3,F4,dF2d\[Mu],dF3d\[Mu],dF4d\[Mu],xQ,\[Zeta]Q,\[CapitalDelta]Bcorr,\[CapitalDelta]Qcorr},(*running coupling and quark masses*)
mubar=4. \[Pi] T Exp[-GammaE+(-Nc+4. Nf Log[4.])/(22. Nc-4. Nf)];
gS2=24. \[Pi]^2/((11. CA-4. TF) Log[mubar/LambdaMS]);
mcor=(Log[muref/LambdaMS]/Log[mubar/LambdaMS])^(9. CF/(11. CA-4. TF));
muCor=muq mcor;mdCor=mdq mcor;
mcCor=mcq mcor;msCor=msq mcor;
xQList={muCor/T,mcCor/T,mdCor/T,msCor/T};
\[Zeta]QList={\[Zeta]u,\[Zeta]c,\[Zeta]d,\[Zeta]s};
Do[xQ=xQList[[q]];\[Zeta]Q=\[Zeta]QList[[q]];
F2=F3=F4=0.;dF2d\[Mu]=dF3d\[Mu]=dF4d\[Mu]=0.;
Do[y1=yMin4+i dy4;
fQ=1./(Exp[Sqrt[y1+xQ^2]-\[Zeta]Q]+1.);
fQbar=1./(Exp[Sqrt[y1+xQ^2]+\[Zeta]Q]+1.);
dfQd\[Mu]=Exp[Sqrt[y1+xQ^2]-\[Zeta]Q]/(Exp[Sqrt[y1+xQ^2]-\[Zeta]Q]+1.)^2;
dfQbard\[Mu]=-Exp[Sqrt[y1+xQ^2]+\[Zeta]Q]/(Exp[Sqrt[y1+xQ^2]+\[Zeta]Q]+1.)^2;
F2+=coeSimps4[[i+1]]*(y1 Sqrt[y1/(y1+xQ^2)])/(8 \[Pi]^2)*(fQ+fQbar);
F3+=-coeSimps4[[i+1]]*(Sqrt[y1/(y1+xQ^2)]/y1)*(fQ+fQbar);
dF2d\[Mu]+=coeSimps4[[i+1]]*(y1 Sqrt[y1/(y1+xQ^2)])/(8 \[Pi]^2)*(dfQd\[Mu]+dfQbard\[Mu]);
dF3d\[Mu]+=-coeSimps4[[i+1]]*(Sqrt[y1/(y1+xQ^2)]/y1)*(dfQd\[Mu]+dfQbard\[Mu]);
Do[y2=yMin4+j dy4;
Module[{fQ2,fQbar2,df2,dfbar2,log1,log2,common},fQ2=1./(Exp[Sqrt[y2+xQ^2]-\[Zeta]Q]+1.);
fQbar2=1./(Exp[Sqrt[y2+xQ^2]+\[Zeta]Q]+1.);
df2=Exp[Sqrt[y2+xQ^2]-\[Zeta]Q]/(Exp[Sqrt[y2+xQ^2]-\[Zeta]Q]+1.)^2;
dfbar2=-Exp[Sqrt[y2+xQ^2]+\[Zeta]Q]/(Exp[Sqrt[y2+xQ^2]+\[Zeta]Q]+1.)^2;
log1=Log[(Sqrt[(y1+xQ^2)(y2+xQ^2)]+xQ^2-Sqrt[y1 y2])/(Sqrt[(y1+xQ^2)(y2+xQ^2)]+xQ^2+Sqrt[y1 y2])];
log2=Log[(Sqrt[(y1+xQ^2)(y2+xQ^2)]-xQ^2+Sqrt[y1 y2])/(Sqrt[(y1+xQ^2)(y2+xQ^2)]-xQ^2-Sqrt[y1 y2]+1.*^-12)];
common=coeSimps4[[i+1]] coeSimps4[[j+1]]/(4 \[Pi])^4/Sqrt[(y1+xQ^2) (y2+xQ^2)];
F4+=common*((fQ fQbar2+fQbar fQ2)*2*log1+(fQ fQ2+fQbar fQbar2)*2*log2);
dF4d\[Mu]+=common*((dfQd\[Mu] fQbar2+fQ dfbar2+dfQbard\[Mu] fQ2+fQbar df2)*log1+(dfQd\[Mu] fQ2+fQ df2+dfQbard\[Mu] fQbar2+fQbar dfbar2)*log2);],{j,0,n4-1}],{i,0,n4-1}];
\[Alpha]E2+=-dA (1/6 F2 (1+6 F2)+(xQ^2/(4 \[Pi]^2)) (3 Log[mubar/(xQ T)]+2) F2-2 xQ^2 F4);
d\[Alpha]E2d\[Mu]+=-dA ((1/6 dF2d\[Mu] (1+6 F2)+F2 dF2d\[Mu])+(xQ^2/(4 \[Pi]^2)) (3 Log[mubar/(xQ T)]+2) dF2d\[Mu]-2 xQ^2 dF4d\[Mu]);
\[Alpha]E7+=2 Log[mubar/(xQ T)]+F3;
d\[Alpha]E7d\[Mu]+=-(2/3) dF3d\[Mu];,{q,1,4}];
\[Alpha]E2-=dA CA/144.;
\[Alpha]E7=22 CA/3 (Log[mubar Exp[GammaE]/(4 \[Pi] T)]+1/22)-(2/3) \[Alpha]E7;
Module[{g32,dg32d\[Mu]},g32=gS2+gS2^2/(4 \[Pi])^2 \[Alpha]E7;
dg32d\[Mu]=gS2^2/(4 \[Pi])^2 d\[Alpha]E7d\[Mu];
\[CapitalDelta]Bcorr=(1/3) (g32 d\[Alpha]E2d\[Mu]+dg32d\[Mu] \[Alpha]E2);
\[CapitalDelta]Qcorr=2/3*(g32 d\[Alpha]E2d\[Mu]+dg32d\[Mu] \[Alpha]E2)   (*u,c*)+-1/3*(g32 d\[Alpha]E2d\[Mu]+dg32d\[Mu] \[Alpha]E2);(*d,s*){\[CapitalDelta]Bcorr,\[CapitalDelta]Qcorr}]],CompilationTarget->"C",RuntimeOptions->"Speed",CompilationOptions->{"InlineExternalDefinitions"->True,"InlineCompiledFunctions"->True}]/. ruleOwn[{GammaE, muq, mdq, mcq, msq,
                Nc, Nf, dA, TF, CA, CF, LambdaMS, muref,  n4, yMin4, dy4, coeSimps4}]// ReleaseHold;
checkIfCompiled[CorAsyQCD]
(*\:25ba 2.3  finite-T hadron/parton DOF  (Bors\[AAcute]nyi et al. fits) ------------*)



(* ::Subsection:: *)
(*Thermodynamics*)


(* ::Subsubsection:: *)
(*SubStar[g]*)


(* ::Input::Initialization:: *)
(*gs*)(*elementary integrals and their derivatives w.r.t.u=m/T*)Sfit[u_]=1+7/4 Exp[-1.0419 u] (1+1.034 u+0.456426 u^2+0.0595249 u^3);
dSfitdx[u_]=D[Sfit[x],x]/. x->u;

frho[u_]=Exp[-1.04855 u] (1+1.03757 u+0.50863 u^2+0.0893988 u^3);
dfrhodx[u_]=D[frho[x],x]/. x->u;

brho[u_]=Exp[-1.03149 u] (1+1.03317 u+0.398264 u^2+0.0648056 u^3);
dbrhodx[u_]=D[brho[x],x]/. x->u;

fs[u_]=Exp[-1.0419 u] (1+1.034 u+0.456426 u^2+0.0506182 u^3);
bs[u_]=Exp[-1.03365 u] (1+1.03397 u+0.342548 u^2+0.0506182 u^3);
(*g\[Rho] and gS below 120 MeV*)
(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)(*SECTION 3 analytic g\[Rho](T),gS(T) fits ( >120 MeV)*)(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)
aVec={1.,1.11724,0.312672,-0.0468049,-0.0265004,-0.0011976,1.82812*^-4,1.36436*^-4,8.55051*^-5,1.2284*^-5,3.82259*^-7,-6.87035*^-9};

bVec={1.43382*^-2,1.37559*^-2,2.92108*^-3,-5.38533*^-4,-1.62496*^-4,-2.87906*^-5,-3.84278*^-6,2.78776*^-6,7.40342*^-7,1.17210*^-7,3.72499*^-9,-6.74107*^-11};

cVec={1.,0.607869,-0.154485,-0.224034,-0.0282147,0.029062,0.00686778,-0.00100005,-1.69104*^-4,1.06301*^-5,1.69528*^-6,-9.33311*^-8};

dVec={70.7388,91.8011,33.1892,-1.39779,-1.52558,-0.0197857,-0.160146,8.22615*^-5,0.0202651,-1.82134*^-5,7.83943*^-5,7.13518*^-5};
(*------------------------------------------------------------------*)(*0. generic Horner evaluator (already fine)*)(*------------------------------------------------------------------*)
polyEvalC=Compile[{{coeff,_Real,1},{t,_Real}},Fold[#2+#1 t&,0.,Reverse[coeff]],CompilationTarget->"C",RuntimeOptions->"Speed"];
checkIfCompiled[polyEvalC]
(*------------------------------------------------------------------*)
(*1. g\[Rho]<120 MeV*)
(*------------------------------------------------------------------*)
grhoBelow120MeV=Hold@Compile[{{T,_Real}},Module[{ue,umu,up0,upc,u1,u2,u3,u4},ue=me/T;
umu=mmu/T;
up0=mpi0/T;upc=mpic/T;
u1=M1/T;u2=M2/T;u3=M3/T;u4=M4/T;
2.030+1.353 Sfit[ue]^(4/3)+3.495 frho[ue]+3.446 frho[umu]+1.05 brho[up0]+2.08 brho[upc]+4.165 brho[u1]+30.55 brho[u2]+89.4 brho[u3]+8209. brho[u4]],CompilationTarget->"C",RuntimeOptions->"Speed",Parallelization->True,CompilationOptions->{"InlineExternalDefinitions"->True,"InlineCompiledFunctions"->True}]/. ruleDown[{Sfit,frho,brho,me,mmu,mpi0,mpic,M1,M2,M3,M4}]//ReleaseHold;
checkIfCompiled[grhoBelow120MeV]
(*------------------------------------------------------------------*)
(*2. gS<120 MeV*)
(*------------------------------------------------------------------*)
gsBelow120MeV=Hold@Compile[{{T,_Real}},Module[{ue,umu,up0,upc,u1,u2,u3,u4},ue=me/T;
umu=mmu/T;
up0=mpi0/T;upc=mpic/T;
u1=M1/T;u2=M2/T;u3=M3/T;u4=M4/T;
2.008+1.923 Sfit[ue]+3.442 fs[ue]+3.468 fs[umu]+1.034 bs[up0]+2.068 bs[upc]+4.16 bs[u1]+30.55 bs[u2]+90. bs[u3]+6209. bs[u4]],CompilationTarget->"C",RuntimeOptions->"Speed",Parallelization->True,CompilationOptions->{"InlineExternalDefinitions"->True,"InlineCompiledFunctions"->True}]/. ruleDown[{Sfit,fs,bs,me,mmu,mpi0,mpic,M1,M2,M3,M4}]//ReleaseHold;
checkIfCompiled[gsBelow120MeV]
(*------------------------------------------------------------------*)
(*3. g\[Rho]\[GreaterEqual]120 MeV (Pad\[EAcute]\[Dash]fit)*)
(*------------------------------------------------------------------*)
grhoAbove120MeV=With[{polyEvalC=polyEvalC},Hold@Compile[{{T,_Real}},Module[{t},t=Log[T*0.001];
polyEvalC[aVec,t]/polyEvalC[bVec,t]],CompilationTarget->"C",RuntimeOptions->"Speed",Parallelization->True,CompilationOptions->{"InlineExternalDefinitions"->True,"InlineCompiledFunctions"->True}]/. ruleOwn[{aVec,bVec}]//ReleaseHold];
checkIfCompiled[grhoAbove120MeV]
(*------------------------------------------------------------------*)
(*4. gS\[GreaterEqual]120 MeV (Pad\[EAcute]\[Dash]fit& extra rational)*)
(*------------------------------------------------------------------*)
gsAbove120MeV=With[{polyEvalC=polyEvalC},Hold@Compile[{{T,_Real}},Module[{t,num,den,ratio},t=Log[T*0.001];
num=polyEvalC[aVec,t];
den=polyEvalC[bVec,t];
ratio=polyEvalC[cVec,t]/polyEvalC[dVec,t];
(num/den)/(1.+ratio)],CompilationTarget->"C",RuntimeOptions->"Speed",Parallelization->True,CompilationOptions->{"InlineExternalDefinitions"->True,"InlineCompiledFunctions"->True}]/. ruleOwn[{aVec,bVec,cVec,dVec}]//ReleaseHold];
checkIfCompiled[gsAbove120MeV]


(* ::Subsubsection:: *)
(*Computing densities, entropies, etc.*)


(* ::Input::Initialization:: *)
(*\:21e9 fully-corrected ThermoQuantitiesC \:21e9*)(* ===============1. ThermoQuantitiesC (pQCD,\[Chi]\:2011window,HG)===============*)(*---------------------------------------------------------------*)(*\[Rho](T,\[Mu]) s(T,\[Mu]) P(T,\[Mu])\[LongDash]compiled C routine*)(*---------------------------------------------------------------*)(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)(*Thermodynamic quantities \[Rho],s,P*)(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)(*\:21e9 fully-corrected ThermoQuantitiesC \:21e9*)(* ===============1. ThermoQuantitiesC (pQCD,\[Chi]\:2011window,HG)===============*)(*---------------------------------------------------------------*)(*\[Rho](T,\[Mu]) s(T,\[Mu]) P(T,\[Mu])\[LongDash]compiled C routine*)(*---------------------------------------------------------------*)(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)(*Thermodynamic quantities \[Rho],s,P*)(*\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)
factorEntropy[y_,EX_,\[Xi]X_]=(y^2*EX+y^4/(3*EX)-\[Xi]X*y^2);
entropySumFD[y_,EX_,\[Xi]X_]=factorEntropy[y,EX,\[Xi]X]*fFD[EX,\[Xi]X]+factorEntropy[y,EX,-\[Xi]X]*fFD[EX,-\[Xi]X];
entropySumBE[y_,EX_,\[Xi]X_]=factorEntropy[y,EX,\[Xi]X]*fBE[EX,\[Xi]X]+factorEntropy[y,EX,-\[Xi]X]*fBE[EX,-\[Xi]X];
fFDsum[EX_,\[Xi]X_]=fFD[EX,\[Xi]X]+fFD[EX,-\[Xi]X];
fFDdiff[EX_,\[Xi]X_]=fFD[EX,\[Xi]X]-fFD[EX,-\[Xi]X];
fBEsum[EX_,\[Xi]X_]=fBE[EX,\[Xi]X]+fBE[EX,-\[Xi]X];
fBEdiff[EX_,\[Xi]X_]=fBE[EX,\[Xi]X]-fBE[EX,-\[Xi]X];
ThermoQuantitiesC=With[{CorAsyQCD=CorAsyQCD,grhoBelow120=grhoBelow120MeV,gsBelow120=gsBelow120MeV,grhoAbove120=grhoAbove120MeV,gsAbove120=gsAbove120MeV,idxBin=findIndexC},Hold@Compile[{{T,_Real},{\[Xi]\[Nu]e,_Real},{\[Xi]\[Nu]\[Mu],_Real},{\[Xi]\[Nu]\[Tau],_Real},{\[Xi]B,_Real},{\[Xi]Q,_Real}},Module[{(*\[HorizontalLine]\[HorizontalLine] chemical potentials \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)\[Xi]e,\[Xi]\[Mu],\[Xi]\[Tau],\[Xi]u,\[Xi]d,\[Xi]c,\[Xi]s,\[Xi]\[Pi]c,\[Xi]p,\[Xi]n,(*m/T ratios*)xe,x\[Mu],x\[Tau],xu,xd,xc,xs,x\[Pi]c,(*energy-density shifts \[CapitalDelta]\[Rho]\[Dash]start at 0.0 automatically*)\[CapitalDelta]\[Rho]\[Nu]\[Alpha]=0.,\[CapitalDelta]\[Rho]\[Nu]\[Mu]=0.,\[CapitalDelta]\[Rho]\[Nu]\[Tau]=0.,\[CapitalDelta]\[Rho]e=0.,\[CapitalDelta]\[Rho]\[Mu]=0.,\[CapitalDelta]\[Rho]\[Tau]=0.,\[CapitalDelta]\[Rho]u=0.,\[CapitalDelta]\[Rho]d=0.,\[CapitalDelta]\[Rho]c=0.,\[CapitalDelta]\[Rho]s=0.,\[CapitalDelta]\[Rho]\[Pi]=0.,(*entropy shifts \[CapitalDelta]s*)s\[Nu]\[Alpha]=0.,\[CapitalDelta]s\[Nu]\[Mu]=0.,\[CapitalDelta]s\[Nu]\[Tau]=0.,\[CapitalDelta]se=0.,\[CapitalDelta]s\[Mu]=0.,\[CapitalDelta]s\[Tau]=0.,\[CapitalDelta]su=0.,\[CapitalDelta]sd=0.,\[CapitalDelta]sc=0.,\[CapitalDelta]ss=0.,\[CapitalDelta]s\[Pi]=0.,(*number-density shifts N\:1d62*)N\[Nu]e=0.,N\[Nu]\[Mu]=0.,N\[Nu]\[Tau]=0.,Ne=0.,N\[Mu]=0.,N\[Tau]=0.,Nu=0.,Nd=0.,Nc=0.,Ns=0.,N\[Pi]=0.,(*asymmetries*)\[CapitalDelta]n\[Nu]\[Alpha]=0.,\[CapitalDelta]N\[Nu]\[Alpha]=0.,\[CapitalDelta]\[ScriptL]=0.,\[CapitalDelta]\[Nu]Tot=0.,\[CapitalDelta]cTot=0.,\[Rho]\[Nu]\[Alpha]=0.,\[Rho]\[ScriptL]=0.,(*lattice/pQCD vars*)\[CapitalDelta]BQCD=0.,\[CapitalDelta]QQCD=0.,NB=0.,NQ=0.,\[Chi]2B=0.,\[Chi]2Q=0.,\[Chi]11=0.,\[Chi]2BS=0.,\[Chi]2QS=0.,\[Chi]11S=0.,(*totals*)s0=0.,\[Rho]0=0.,\[CapitalDelta]\[Rho]=0.,s=0.,\[Rho]=0.,P=0.,(*helpers*)fps=1./(2. Pi^2),nG=n3,i,y,w,ws,Ee,E\[Mu],E\[Tau],Eu=0.,Ed=0.,Ec=0.,Es=0.,E\[Pi]=0.,\[CapitalDelta]lTot=0.,\[CapitalDelta]NlTot=0.,\[CapitalDelta]Nl\[Alpha]=0.,\[CapitalDelta]N\[Nu]Tot=0.,N\[Nu]\[Alpha]=0.,s\[Nu]\[Alpha]Corr=0.,\[Rho]\[Nu]\[Alpha]Corr=0.,N\[Nu]\[Alpha]Corr=0.,\[Rho]\[Nu]Tot=0.,\[Rho]\[Nu]TotCorr=0.,N\[Nu]Tot=0.,N\[Nu]TotCorr=0.,summ=0.,\[CapitalDelta]s\[Nu]Tot=0.,\[CapitalDelta]s\[Nu]TotCorr=0.,Nl\[Alpha]=0.,Nl\[Alpha]Corr=0.,NlTot=0.,NlTotCorr=0.,\[Rho]l\[Alpha]=0.,\[CapitalDelta]\[Rho]l\[Alpha]=0.,\[Rho]l\[Alpha]Corr=0.,\[Rho]lTot=0.,\[Rho]lTotCorr=0.,\[CapitalDelta]NqTot=0.,NqTot=0.,NqTotCorr=0.,\[Rho]qTot=0.,\[Rho]qTotCorr=0.,\[Rho]\[Pi]=0.,\[Rho]\[Pi]Corr=0.,N\[Pi]Corr=0.,\[CapitalDelta]N\[Pi]=0.,sl\[Alpha]=0.,slTot=0.,sl\[Alpha]Corr=0.,slTotCorr=0.,s\[Nu]Tot=0.,s\[Nu]TotCorr=0.,sqcdTot=0.,sqcdTotCorr=0.,s\[Pi]=0.,s\[Pi]Corr=0.,\[CapitalDelta]Ql=0.,\[CapitalDelta]Qq=0.,\[CapitalDelta]Q\[Pi]=0.,\[CapitalDelta]Q=0.,\[CapitalDelta]QB=0.,\[CapitalDelta]Ql\[Alpha]=0.,\[CapitalDelta]Qstrong=0.,Nl\[Mu]Corr=0.,Nl\[Tau]Corr=0.,N\[Nu]\[Mu]Corr=0.,N\[Nu]\[Tau]Corr=0.,NuCorr=0.,NdCorr=0.,NsCorr=0.,NcCorr=0.,\[Rho]l\[Nu]Corr=0.,sl\[Nu]Corr=0.,pl\[Nu]Corr=0.,xp=mp/T,xn=mn/T,\[Rho]TotCorr=0.,pqcdTotCorr=0.,\[Rho]qcdTotCorr=0.,\[Rho]qcdTot=0.,\[CapitalDelta]Nl\[Mu]=0.,\[CapitalDelta]Nl\[Tau]=0.,\[CapitalDelta]N\[Nu]\[Mu]=0.,\[CapitalDelta]N\[Nu]\[Tau]=0.,\[CapitalDelta]Nu=0.,\[CapitalDelta]Nd=0.,\[CapitalDelta]Ns=0.,\[CapitalDelta]Nc=0.,Nl\[Mu]=0.,Nl\[Tau]=0.},(*\[HorizontalLine]\[HorizontalLine] 1. bookkeeping for \[Mu]/T values \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)
{\[Xi]e,\[Xi]\[Mu],\[Xi]\[Tau]}={\[Xi]\[Nu]e-\[Xi]Q,\[Xi]\[Nu]\[Mu]-\[Xi]Q,\[Xi]\[Nu]\[Tau]-\[Xi]Q};
{\[Xi]u,\[Xi]d}={\[Xi]B/3.+2. \[Xi]Q/3.,\[Xi]B/3.-\[Xi]Q/3.};
\[Xi]c=\[Xi]u;
\[Xi]s=\[Xi]d;
\[Xi]\[Pi]c=-\[Xi]Q;
\[Xi]p=\[Xi]B/3.+\[Xi]Q;
\[Xi]n=\[Xi]B/3.;
xe=me/T;x\[Mu]=mmu/T;x\[Tau]=mtau/T;
xu=muq/T;xd=mdq/T;xc=mcq/T;xs=msq/T;
x\[Pi]c=mpic/T;
(*\[HorizontalLine]\[HorizontalLine] 3. Simpson 41-point integration (unchanged except \[Pi]-block) \[HorizontalLine]*)
For[i=0,i<nG,i++,y=yMin3+i dy3;
w=coeSimps3[[i+1]] y^2;
ws=coeSimps3[[i+1]];
(*-----electron neutrinos------------------------------------------------*)
\[CapitalDelta]N\[Nu]\[Alpha]+=w *fFDdiff[y,\[Xi]\[Nu]e];
\[CapitalDelta]N\[Nu]\[Mu]+=w *fFDdiff[y,\[Xi]\[Nu]\[Mu]];
\[CapitalDelta]N\[Nu]\[Tau]+=w *fFDdiff[y,\[Xi]\[Nu]\[Tau]];
summ=w* y *fFDsum[y,\[Xi]\[Nu]e];
\[Rho]\[Nu]\[Alpha]+=summ;
\[Rho]\[Nu]\[Alpha]Corr+=summ-w*y*fFDsum[y,0.];
summ=w *fFDsum[y,\[Xi]\[Nu]e];
N\[Nu]\[Alpha]+=summ;
N\[Nu]\[Alpha]Corr+=summ-w*fFDsum[y,0.];
summ=w *fFDsum[y,\[Xi]\[Nu]\[Mu]];
N\[Nu]\[Mu]+=summ;
N\[Nu]\[Mu]Corr+=summ-w*fFDsum[y,0.];
summ=w*fFDsum[y,\[Xi]\[Nu]\[Tau]];
N\[Nu]\[Tau]+=summ;
N\[Nu]\[Tau]Corr+=summ-w*fFDsum[y,0.];
summ=ws*entropySumFD[y,y,\[Xi]\[Nu]e];
s\[Nu]\[Alpha]+=summ;
s\[Nu]\[Alpha]Corr+=summ-ws*entropySumFD[y,y,0.];
(*------------Neutrinos - total--------------*)
\[CapitalDelta]N\[Nu]Tot+=w (fFDdiff[y,\[Xi]\[Nu]e]+fFDdiff[y,\[Xi]\[Nu]\[Mu]]+fFDdiff[y,\[Xi]\[Nu]\[Tau]]);
summ=w (fFDsum[y,\[Xi]\[Nu]e]+fFDsum[y,\[Xi]\[Nu]\[Mu]]+fFDsum[y,\[Xi]\[Nu]\[Tau]]);
N\[Nu]Tot+=summ;
N\[Nu]TotCorr+=summ-3*w*fFDsum[y,0.];
summ=w*y*(fFDsum[y,\[Xi]\[Nu]e]+fFDsum[y,\[Xi]\[Nu]\[Mu]]+fFDsum[y,\[Xi]\[Nu]\[Tau]]);
\[Rho]\[Nu]Tot+=summ;
\[Rho]\[Nu]TotCorr+=summ-3*w*y*fFDsum[y,0.];
(*Entropy*)
summ=ws (entropySumFD[y,y,\[Xi]\[Nu]e]+entropySumFD[y,y,\[Xi]\[Nu]\[Mu]]+entropySumFD[y,y,\[Xi]\[Nu]\[Tau]]);
s\[Nu]Tot+=summ;
s\[Nu]TotCorr+=summ-3*ws* entropySumFD[y,y,0.];
(*------------Charged leptons - electrons and total------------*)
Ee=Sqrt[y^2+xe^2];E\[Mu]=Sqrt[y^2+x\[Mu]^2];E\[Tau]=Sqrt[y^2+x\[Tau]^2];
summ=w *fFDdiff[Ee,\[Xi]e];
\[CapitalDelta]Nl\[Alpha]+=summ;
summ=w*fFDdiff[E\[Mu],\[Xi]\[Mu]];
\[CapitalDelta]Nl\[Mu]+=summ;
summ=w*fFDdiff[E\[Tau],\[Xi]\[Tau]];
\[CapitalDelta]Nl\[Tau]+=summ;
summ=w*fFDsum[Ee,\[Xi]e];
Nl\[Alpha]+=summ;
Nl\[Alpha]Corr+=summ-w*fFDsum[Ee,0.];
summ=w*fFDsum[E\[Mu],\[Xi]\[Mu]];
Nl\[Mu]+=summ;
Nl\[Mu]Corr+=summ-w*fFDsum[E\[Mu],0.];
summ=w*fFDsum[E\[Tau],\[Xi]\[Tau]];
Nl\[Tau]+=summ;
Nl\[Tau]Corr+=summ-w*fFDsum[E\[Tau],0.];
summ=w*(fFDdiff[Ee,\[Xi]e]+fFDdiff[E\[Mu],\[Xi]\[Mu]]+fFDdiff[E\[Tau],\[Xi]\[Tau]]);
\[CapitalDelta]NlTot+=summ;
\[CapitalDelta]Ql+=summ;
summ=w*(fFDsum[Ee,\[Xi]e]+fFDsum[E\[Mu],\[Xi]\[Mu]]+fFDsum[E\[Tau],\[Xi]\[Tau]]);
NlTot+=summ;
NlTotCorr+=summ-w*(fFDsum[Ee,0.]+fFDsum[E\[Mu],0.]+fFDsum[E\[Tau],0.]);
(*Energy densities*)
summ=w*Ee*fFDsum[Ee,\[Xi]e];
\[Rho]l\[Alpha]+=summ;
\[Rho]l\[Alpha]Corr+=summ-w*Ee*fFDsum[Ee,0.];
summ=w*(Ee*fFDsum[Ee,\[Xi]e]+E\[Mu]*fFDsum[E\[Mu],\[Xi]\[Mu]]+E\[Tau]*fFDsum[E\[Tau],\[Xi]\[Tau]]);
\[Rho]lTot+=summ;
\[Rho]lTotCorr+=summ-w*(Ee*fFDsum[Ee,0.]+E\[Mu]*fFDsum[E\[Mu],0.]+E\[Tau]*fFDsum[E\[Tau],0.]);
(*Entropy*)
summ=ws*entropySumFD[y,Ee,\[Xi]e];
sl\[Alpha]+=summ;
sl\[Alpha]Corr+=summ-entropySumFD[y,Ee,0.];
summ=ws*(entropySumFD[y,Ee,\[Xi]e]+entropySumFD[y,E\[Mu],\[Xi]\[Mu]]+entropySumFD[y,E\[Tau],\[Xi]\[Tau]]);
slTot+=summ;
slTotCorr+=summ-ws*(entropySumFD[y,Ee,0.]+entropySumFD[y,E\[Mu],0.]+entropySumFD[y,E\[Tau],0.]);
If[T>280.,
(*-----light quarks (T>120 MeV)--------------------------------*)
Eu=Sqrt[y^2+xu^2];Ed=Sqrt[y^2+xd^2];Ec=Sqrt[y^2+xc^2];Es=Sqrt[y^2+xs^2];
\[CapitalDelta]NqTot+=w*(fFDdiff[Eu,\[Xi]u]+fFDdiff[Ed,\[Xi]d]+fFDdiff[Es,\[Xi]s]+fFDdiff[Ec,\[Xi]c]);
\[CapitalDelta]Nu+=w*fFDdiff[Eu,\[Xi]u];
\[CapitalDelta]Nd+=w*fFDdiff[Ed,\[Xi]d];
\[CapitalDelta]Ns+=w*fFDdiff[Es,\[Xi]s];
\[CapitalDelta]Nc+=w*fFDdiff[Ec,\[Xi]c];
\[CapitalDelta]Qstrong+=w*(2/3 fFDdiff[Eu,\[Xi]u]-1/3 fFDdiff[Ed,\[Xi]d]-1/3 fFDdiff[Es,\[Xi]s]+2/3 fFDdiff[Ec,\[Xi]c]);
summ=w*(fFDsum[Eu,\[Xi]u]+fFDsum[Ed,\[Xi]d]+fFDsum[Es,\[Xi]s]+fFDsum[Ec,\[Xi]c]);
NqTot+=summ;
NqTotCorr+=summ-w*(fFDsum[Eu,0.]+fFDsum[Ed,0.]+fFDsum[Es,0.]+fFDsum[Ec,0.]);
summ=w*fFDsum[Eu,\[Xi]u];
Nu+=summ;
NuCorr+=summ-w*fFDsum[Eu,0.];
summ=w*fFDsum[Ed,\[Xi]d];
Nd+=summ;
NdCorr+=summ-w*fFDsum[Ed,\[Xi]d];
summ=w*fFDsum[Es,\[Xi]s];
Ns+=summ;
NsCorr+=summ-w*fFDsum[Es,\[Xi]s];
summ=w*fFDsum[Ec,\[Xi]c];
Nc+=summ;
NcCorr+=summ-w*fFDsum[Ec,\[Xi]c];
summ=w*(Eu*fFDsum[Eu,\[Xi]u]+Ed*fFDsum[Ed,\[Xi]d]+Es*fFDsum[Es,\[Xi]s]+Ec*fFDsum[Ec,\[Xi]c]);
\[Rho]qcdTot+=summ;
\[Rho]qcdTotCorr+=summ-w*(Eu*fFDsum[Eu,0.]+Ed*fFDsum[Ed,0.]+Es*fFDsum[Es,0.]+Ec*fFDsum[Ec,0.]);
(*Entropy*)
summ=ws*(entropySumFD[y,Eu,\[Xi]u]+entropySumFD[y,Ed,\[Xi]d]+entropySumFD[y,Es,\[Xi]s]+entropySumFD[y,Ec,\[Xi]c]);
sqcdTot+=summ;
sqcdTotCorr+=summ-ws*(entropySumFD[y,Eu,0.]+entropySumFD[y,Ed,0.]+entropySumFD[y,Es,0.]+entropySumFD[y,Ec,0.]);
];
If[T<120.,
E\[Pi]=Sqrt[y^2+x\[Pi]c^2];
(*-----charged \[Pi]\[PlusMinus](only affect low-T)--------------------------*)
summ=w *fBEdiff[E\[Pi],\[Xi]\[Pi]c];
\[CapitalDelta]N\[Pi]+=summ;
\[CapitalDelta]Qstrong-=summ;
summ=w *fBEsum[E\[Pi],\[Xi]\[Pi]c];
N\[Pi]+=summ;
N\[Pi]Corr+=summ-w*fBEsum[E\[Pi],0.];
summ=w* E\[Pi]*fBEsum[E\[Pi],\[Xi]\[Pi]c];
\[Rho]qcdTot+=summ;
\[Rho]qcdTotCorr+=summ-w*E\[Pi]*fBEsum[E\[Pi],0.];
summ=ws*entropySumBE[y,E\[Pi],\[Xi]\[Pi]c];
sqcdTot+=summ;
sqcdTotCorr+=summ-ws*entropySumBE[y,E\[Pi],0.];
];
];
(*\[HorizontalLine]\[HorizontalLine] 4. common 1/(2\[Pi]\.b2) prefactors \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)
(*Neutrinos*)
N\[Nu]\[Alpha]*=fps; N\[Nu]\[Alpha]Corr*=fps; \[CapitalDelta]N\[Nu]\[Alpha]*=fps;
N\[Nu]\[Mu]*=fps;N\[Nu]\[Tau]*=fps;\[CapitalDelta]N\[Nu]\[Mu]*=fps;\[CapitalDelta]N\[Nu]\[Tau]*=fps;N\[Nu]\[Mu]Corr*=fps; N\[Nu]\[Tau]Corr*=fps;
N\[Nu]Tot*=fps; N\[Nu]TotCorr*=fps; 
\[Rho]\[Nu]\[Alpha]*=fps;\[Rho]\[Nu]\[Alpha]Corr*=fps;\[CapitalDelta]\[Rho]\[Nu]\[Alpha]*=fps;\[Rho]\[Nu]Tot*=fps;\[Rho]\[Nu]TotCorr*=fps;
s\[Nu]\[Alpha]*=fps;s\[Nu]Tot*=fps;s\[Nu]\[Alpha]Corr*=fps;s\[Nu]TotCorr*=fps;
(*Charged leptons*)
Nl\[Alpha]*=2*fps; Nl\[Alpha]Corr*=2*fps; \[CapitalDelta]Nl\[Alpha]*=2*fps;
Nl\[Mu]*=2*fps; Nl\[Mu]Corr*=2*fps; \[CapitalDelta]Nl\[Mu]*=2*fps;Nl\[Tau]*=2*fps; Nl\[Tau]Corr*=2*fps; \[CapitalDelta]Nl\[Tau]*=2*fps;
NlTot*=2*fps; NlTotCorr*=2*fps; Nl\[Mu]Corr*=2*fps; Nl\[Tau]Corr*=2*fps;
\[CapitalDelta]Ql*=2*fps;\[CapitalDelta]Ql\[Alpha]*=2*fps;
\[Rho]l\[Alpha]*=2*fps;\[Rho]l\[Alpha]Corr*=2*fps;\[CapitalDelta]\[Rho]l\[Alpha]*=2*fps;\[Rho]lTot*=2*fps;\[Rho]lTotCorr*=2*fps;
sl\[Alpha]*=2*fps;slTot*=2*fps;sl\[Alpha]Corr*=2*fps;slTotCorr*=2*fps;
If[T>280.,
(*Quarks*)
NqTot*=6. fps; NqTotCorr*=6. fps; \[CapitalDelta]NqTot*=6. fps; 
Nu*=6*fps; Nd*=6*fps; Ns*=6*fps; Nc*=6*fps; 
NuCorr*=6*fps; NdCorr*=6*fps; NsCorr*=6*fps; NcCorr*=6*fps; 
\[CapitalDelta]Nu*=6. fps;\[CapitalDelta]Nd*=6. fps;\[CapitalDelta]Nc*=6. fps;\[CapitalDelta]Ns*=6. fps;
\[CapitalDelta]Qstrong*=6*fps;
\[Rho]qcdTot*=6. fps;\[Rho]qcdTotCorr*=6. fps;
sqcdTot*=6*fps;sqcdTotCorr*=6*fps;
pqcdTotCorr=\[Xi]u*\[CapitalDelta]Nu+\[Xi]d*\[CapitalDelta]Nd+\[Xi]s*\[CapitalDelta]Ns+\[Xi]c*\[CapitalDelta]Nc;
];
If[T<120.,
(*Pions*)
N\[Pi]*=fps;N\[Pi]Corr*=fps;
\[Rho]qcdTot*=fps;\[Rho]qcdTotCorr*=fps;
sqcdTot*=fps;sqcdTotCorr*=fps;
\[CapitalDelta]Qstrong*=fps;\[CapitalDelta]N\[Pi]*=fps;
pqcdTotCorr=\[Xi]\[Pi]c*\[CapitalDelta]N\[Pi];
];
If[120<=T<=280,
Module[{idx},
idx=idxBin[T,tGrid,Length[tGrid]];
\[Chi]2B=\[Chi]2Bgrid[[idx]]+(T-tGrid[[idx]]) s2B[[idx]];
\[Chi]2Q=\[Chi]2Qgrid[[idx]]+(T-tGrid[[idx]]) s2Q[[idx]];
\[Chi]11=\[Chi]11grid[[idx]]+(T-tGrid[[idx]]) s11[[idx]];
\[Chi]2BS=s2B[[idx]];
\[Chi]2QS=s2Q[[idx]];
\[Chi]11S=s11[[idx]];
];
NB=\[Chi]11 \[Xi]Q+\[Chi]2B \[Xi]B;
NQ=\[Chi]11 \[Xi]B+\[Chi]2Q \[Xi]Q;
\[CapitalDelta]Qstrong=NQ;
sqcdTotCorr=1/2*T*(\[Chi]2BS*\[Xi]B^2+2. \[Chi]11S*\[Xi]B*\[Xi]Q+\[Chi]2QS*\[Xi]Q^2);
pqcdTotCorr=\[Xi]B*NB+\[Xi]Q*NQ;
\[Rho]qcdTotCorr=1/2*(\[Chi]2B \[Xi]B^2+2. \[Chi]11*\[Xi]B*\[Xi]Q+\[Chi]2Q*\[Xi]Q^2)+1/2 T (\[Chi]2BS*\[Xi]B^2+2. \[Chi]11S*\[Xi]B*\[Xi]Q+\[Chi]2QS*\[Xi]Q^2);
];
(*\[HorizontalLine]\[HorizontalLine] 5. \[Mu]=0 baseline (unchanged) \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)
s0=(2Pi^2)/45*If[T<120.,gsBelow120[T],gsAbove120[T]];
\[Rho]0=Pi^2/30*If[T<120.,grhoBelow120[T],grhoAbove120[T]];
\[Rho]l\[Nu]Corr=\[Rho]\[Nu]TotCorr+\[Rho]lTotCorr;
sl\[Nu]Corr=s\[Nu]TotCorr+slTotCorr;
pl\[Nu]Corr=(\[Xi]\[Nu]e*\[CapitalDelta]N\[Nu]\[Alpha]+\[Xi]\[Nu]\[Mu]*\[CapitalDelta]N\[Nu]\[Mu]+\[Xi]\[Nu]\[Tau]*\[CapitalDelta]N\[Nu]\[Tau]+\[Xi]e*\[CapitalDelta]Nl\[Alpha]+\[Xi]\[Mu]*\[CapitalDelta]Nl\[Mu]+\[Xi]\[Tau]*\[CapitalDelta]Nl\[Tau]);
\[Rho]TotCorr=\[Rho]l\[Nu]Corr+\[Rho]qcdTotCorr;
\[Rho]=\[Rho]0+\[Rho]TotCorr;
s=s0+sl\[Nu]Corr+sqcdTotCorr;
P=s-\[Rho]+pl\[Nu]Corr+pqcdTotCorr;
(*\[HorizontalLine]\[HorizontalLine] 7. return vector (order unchanged) \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]*)
{\[Rho],s,P,\[Rho]TotCorr,\[Rho]\[Nu]\[Alpha],\[Rho]l\[Alpha],\[CapitalDelta]N\[Nu]\[Alpha],\[CapitalDelta]N\[Nu]\[Mu],\[CapitalDelta]N\[Nu]\[Tau],\[CapitalDelta]N\[Nu]Tot,\[CapitalDelta]Nl\[Alpha],\[CapitalDelta]Nl\[Mu],\[CapitalDelta]Nl\[Tau],\[CapitalDelta]NlTot,\[CapitalDelta]Nu,\[CapitalDelta]Nd,\[CapitalDelta]Ns,\[CapitalDelta]Nc,\[CapitalDelta]N\[Pi],\[CapitalDelta]Qstrong,\[Chi]2B,\[Chi]2Q,\[Chi]11,Nl\[Alpha],Nl\[Mu],Nl\[Tau],N\[Nu]\[Alpha],N\[Nu]\[Mu],N\[Nu]\[Tau],\[Xi]\[Pi]c,\[CapitalDelta]N\[Pi]}
],(*end Module*)CompilationTarget->"C",RuntimeOptions->"Speed",CompilationOptions->{"InlineExternalDefinitions"->True,"InlineCompiledFunctions"->True}]/. ruleOwn[{me,mmu,mtau,muq,mdq,mcq,msq,mp,mn,mpic,n3,yMin3,dy3,coeSimps3,tGrid,\[Chi]2Bgrid,\[Chi]2Qgrid,\[Chi]11grid,s2B,s2Q,s11,Tcut,mpl}]/. ruleDown[{fFD,fBE,entropySumFD,entropySumBE,fFDdiff,fFDsum,fBEdiff,fBEsum}]//ReleaseHold];
checkIfCompiled[ThermoQuantitiesC]
ThermoQuantitiesC[100.,2.35,-2.38,0.,0.,0.66]
ThermoQuantitiesC[3000.,0,0.,0.,0.,0.]
Print["Cross-check - Gibbs identity:"]
val0=3.94;
{\[Mu]T\[Nu]ev,\[Mu]T\[Nu]\[Mu]v,\[Mu]T\[Nu]\[Tau]v,\[Mu]TBv,\[Mu]TCv}={val0,val0,0.,0.,0.66};
{\[Mu]Tev,\[Mu]T\[Mu]v,\[Mu]T\[Tau]v}={\[Mu]T\[Nu]ev,\[Mu]T\[Nu]\[Mu]v,\[Mu]T\[Nu]\[Tau]v}-\[Mu]TCv;
Tv=15.;
datt=ThermoQuantitiesC[Tv,\[Mu]T\[Nu]ev,\[Mu]T\[Nu]\[Mu]v,\[Mu]T\[Nu]\[Tau]v,\[Mu]TBv,\[Mu]TCv];
{\[Rho]v,sv,pv}={Tv^4,Tv^3,Tv^4}*Take[datt,{1,3}];
{\[CapitalDelta]nev,\[CapitalDelta]n\[Mu]v,\[CapitalDelta]n\[Tau]v,\[CapitalDelta]nvev,\[CapitalDelta]n\[Nu]\[Mu]v,\[CapitalDelta]n\[Nu]\[Tau]v,\[CapitalDelta]n\[Pi]}=Tv^3*datt[[#]]&/@{7,8,9,11,12,13,-1};
\[Mu]T\[Pi]=datt[[-1]];
{\[Mu]\[Nu]ev,\[Mu]\[Nu]\[Mu]v,\[Mu]\[Nu]\[Tau]v,\[Mu]ev,\[Mu]\[Mu]v,\[Mu]\[Tau]v,\[Mu]\[Pi]}=Tv{\[Mu]T\[Nu]ev,\[Mu]T\[Nu]\[Mu]v,\[Mu]T\[Nu]\[Tau]v,\[Mu]Tev,\[Mu]T\[Mu]v,\[Mu]T\[Tau]v,\[Mu]T\[Pi]};
(pv+\[Rho]v-Total[{\[Mu]\[Nu]ev,\[Mu]\[Nu]\[Mu]v,\[Mu]\[Nu]\[Tau]v,\[Mu]ev,\[Mu]\[Mu]v,\[Mu]\[Tau]v,\[Mu]\[Pi]}*{\[CapitalDelta]nev,\[CapitalDelta]n\[Mu]v,\[CapitalDelta]n\[Tau]v,\[CapitalDelta]nvev,\[CapitalDelta]n\[Nu]\[Mu]v,\[CapitalDelta]n\[Nu]\[Tau]v,\[CapitalDelta]n\[Pi]}])/(Tv*sv)


(* ::Subsection:: *)
(*Solver for the asymmetries*)


(* ::Input::Initialization:: *)
fVecC=With[{TQ=ThermoQuantitiesC},Hold@Compile[{{t,_Real},(*temperature*){\[Zeta],_Real,1},{Le,_Real},{L\[Mu],_Real},{L\[Tau],_Real}},(*{\[Zeta]\[Nu]e,\[Zeta]\[Nu]\[Micro],\[Zeta]\[Nu]\[Tau],\[Zeta]B,\[Zeta]Q}*)Module[{\[Zeta]\[Nu]e=\[Zeta][[1]],\[Zeta]\[Nu]\[Micro]=\[Zeta][[2]],\[Zeta]\[Nu]\[Tau]=\[Zeta][[3]],\[Zeta]B=\[Zeta][[4]],\[Zeta]Q=\[Zeta][[5]],out,(*ThermoQuantitiesC result*)\[CapitalDelta]\[Nu]e,\[CapitalDelta]\[Nu]\[Micro],\[CapitalDelta]\[Nu]\[Tau],\[CapitalDelta]e,\[CapitalDelta]\[Micro],\[CapitalDelta]\[Tau],\[CapitalDelta]u,\[CapitalDelta]d,\[CapitalDelta]c,\[CapitalDelta]s,\[CapitalDelta]\[Pi],NB,NQ,s},(*1. take everything from the big thermo kernel---------------*)out=TQ[t,\[Zeta]\[Nu]e,\[Zeta]\[Nu]\[Micro],\[Zeta]\[Nu]\[Tau],\[Zeta]B,\[Zeta]Q];
s=out[[2]];(*entropy density*)\[CapitalDelta]\[Nu]e=out[[7]];\[CapitalDelta]\[Nu]\[Micro]=out[[8]];\[CapitalDelta]\[Nu]\[Tau]=out[[9]];
\[CapitalDelta]e=out[[11]];\[CapitalDelta]\[Micro]=out[[12]];\[CapitalDelta]\[Tau]=out[[13]];
\[CapitalDelta]u=out[[15]];\[CapitalDelta]d=out[[16]];\[CapitalDelta]s=out[[17]];\[CapitalDelta]c=out[[18]];
\[CapitalDelta]\[Pi]=out[[19]];
(*2. Baryon& charge densities,exactly as in the old code---*)Which[t>=Tcut,(*pQCD--------------*)NB=(\[CapitalDelta]u+\[CapitalDelta]d+\[CapitalDelta]c+\[CapitalDelta]s)/3.;
NQ=-(\[CapitalDelta]e+\[CapitalDelta]\[Micro]+\[CapitalDelta]\[Tau])+2/3(\[CapitalDelta]u+\[CapitalDelta]c)-1/3(\[CapitalDelta]d+\[CapitalDelta]s),120.<=t<Tcut,(*\[Chi]-window----------*)With[{\[Chi]11=out[[23]],\[Chi]2B=out[[21]],\[Chi]2Q=out[[22]]},NB=\[Chi]11 \[Zeta]Q+\[Chi]2B \[Zeta]B;
NQ=\[Chi]11 \[Zeta]B+\[Chi]2Q \[Zeta]Q-(\[CapitalDelta]e+\[CapitalDelta]\[Micro]+\[CapitalDelta]\[Tau])],True,(*hadron gas--------*)NB=0.;(*p& n==0 in this model*)
NQ=-(\[CapitalDelta]e+\[CapitalDelta]\[Micro]+\[CapitalDelta]\[Tau]+\[CapitalDelta]\[Pi])];
(*3. return the five residuals-------------------------------*){(\[CapitalDelta]\[Nu]e+\[CapitalDelta]e)-Le s,(\[CapitalDelta]\[Nu]\[Micro]+\[CapitalDelta]\[Micro])-L\[Mu] s,(\[CapitalDelta]\[Nu]\[Tau]+\[CapitalDelta]\[Tau])-L\[Tau] s,NB-\[Eta]B s,NQ}],CompilationTarget->"C",RuntimeOptions->"Speed"]/.ruleOwn[{Tcut}]//ReleaseHold]
thermoR[T_?NumericQ,\[Zeta]\[Nu]e_?NumericQ,\[Zeta]\[Nu]\[Mu]_?NumericQ,\[Zeta]\[Nu]\[Tau]_?NumericQ,\[Zeta]B_?NumericQ,\[Zeta]Q_?NumericQ,Le_,L\[Mu]_,L\[Tau]_]:=With[{t=N[T,16],z1=N[\[Zeta]\[Nu]e,16],z2=N[\[Zeta]\[Nu]\[Mu],16],z3=N[\[Zeta]\[Nu]\[Tau],16],z4=N[\[Zeta]B,16],z5=N[\[Zeta]Q,16]},fVecC[t,{z1,z2,z3,z4,z5},Le,L\[Mu],L\[Tau]]];
thermoR4[T_?NumericQ,\[Zeta]\[Nu]e_?NumericQ,\[Zeta]\[Nu]\[Mu]_?NumericQ,\[Zeta]\[Nu]\[Tau]_?NumericQ,\[Zeta]B_?NumericQ,\[Zeta]Q_?NumericQ,Le_,L\[Mu]_,L\[Tau]_]:=With[{t=N[T,16],z1=N[\[Zeta]\[Nu]e,16],z2=N[\[Zeta]\[Nu]\[Mu],16],z3=N[\[Zeta]\[Nu]\[Tau],16],z4=N[\[Zeta]B,16],z5=N[\[Zeta]Q,16]},Delete[fVecC[t,{z1,z2,z3,z4,z5},Le,L\[Mu],L\[Tau]],{4}]];
SolverAsymmetriesNoNB[Tmin_,Tmax_,Le_,L\[Mu]_,L\[Tau]_,\[CapitalDelta]val_]:=Block[{},
Tvals=Select[Join[Table[10^t,{t,Log10[Tmin],Log10[Tmax],(Log10[Tmax]-Log10[Tmin])/100.}],{114.,116.}]//Sort//DeleteDuplicates//N//Reverse,!(115<#<125)&];
Tvals=Tvals-\[CapitalDelta]val;
TabRes={};
{\[Xi]1start,\[Xi]2start,\[Xi]3start,\[Xi]5start}=ConstantArray[0.,4];
Do[
sols={\[Xi]1,\[Xi]2,\[Xi]3,\[Xi]5}/.FindRoot[thermoR4[T,\[Xi]1,\[Xi]2,\[Xi]3,0.,\[Xi]5,Le,L\[Mu],L\[Tau]]==0.,{{\[Xi]1,\[Xi]1start},{\[Xi]2,\[Xi]2start},{\[Xi]3,\[Xi]3start},{\[Xi]5,\[Xi]5start}},MaxIterations->150,Method->{"Newton"}];
{\[Xi]1start,\[Xi]2start,\[Xi]3start,\[Xi]5start}=sols;
TabRes=Join[TabRes,{{T,\[Xi]1start,\[Xi]2start,\[Xi]3start,0.,\[Xi]5start}}];
,{T,Tvals}];
{\[Mu]behavior["ve",Le,L\[Mu],L\[Tau]],\[Mu]behavior["vmu",Le,L\[Mu],L\[Tau]],\[Mu]behavior["vtau",Le,L\[Mu],L\[Tau]]}=TabRes[[All,{1,#}]]&/@Range[2,4];
{\[Mu]behavior["e",Le,L\[Mu],L\[Tau]],\[Mu]behavior["mu",Le,L\[Mu],L\[Tau]],\[Mu]behavior["tau",Le,L\[Mu],L\[Tau]]}=Table[{#[[1]],#[[x]]-#[[6]]}&/@TabRes,{x,Range[2,4]}];
TabRes
]


(* ::Subsubsection:: *)
(*Example*)


(* ::Input::Initialization:: *)
{Let,L\[Mu]t,L\[Tau]t}={0.1,-0.1,0.}//N;
SolverAsymmetriesNoNB[10,10^4,0.1,-0.1,0.,0.];
ListLogLinearPlot[{\[Mu]behavior["ve",Let,L\[Mu]t,L\[Tau]t],\[Mu]behavior["vmu",Let,L\[Mu]t,L\[Tau]t],\[Mu]behavior["vtau",Let,L\[Mu]t,L\[Tau]t],\[Mu]behavior["e",Let,L\[Mu]t,L\[Tau]t],\[Mu]behavior["mu",Let,L\[Mu]t,L\[Tau]t],\[Mu]behavior["tau",Let,L\[Mu]t,L\[Tau]t]},Joined->True,ImageSize->Large,Frame->True,FrameStyle->Directive[Thick,Black,20],PlotStyle->Join[{Thick,#}&/@{Darker@Red,Blue,Green},{Thick,#,Dashing[0.01]}&/@{Darker@Red,Blue,Green}],PlotLegends->Placed[Style[Row[{#}],20,Black]&/@{"\!\(\*SubscriptBox[\(\[Nu]\), \(e\)]\)","\!\(\*SubscriptBox[\(\[Nu]\), \(\[Mu]\)]\)","\!\(\*SubscriptBox[\(\[Nu]\), \(\[Tau]\)]\)","e","\[Mu]","\[Tau]"},{0.9,0.7}],PlotRange->{{10,10^4},All},PlotLabel->Style[Row[{"\!\(\*SubscriptBox[\(L\), \(e\)]\) = " ,Let,", \!\(\*SubscriptBox[\(L\), \(\[Mu]\)]\) = ",L\[Mu]t,", \!\(\*SubscriptBox[\(L\), \(\[Tau]\)]\) = ",L\[Tau]t}],20,Black],FrameLabel->{"T [MeV]","\!\(\*SubscriptBox[\(\[Xi]\), \(x\)]\)"}]



(* ::Input:: *)
(**)


(* ::Subsubsection::Closed:: *)
(*Old*)


(* ::Input:: *)
(*IfTest=False;*)
(*If[IfTest,*)
(*asymmetryC=With[{idx=findIndexC},Hold@Compile[{{T,_Real},{\[Zeta]\[Nu]e,_Real},{\[Zeta]\[Nu]\[Mu],_Real},{\[Zeta]\[Nu]\[Tau],_Real},{\[Zeta]B,_Real},{\[Zeta]Q,_Real},{dummy,_Integer}},Module[{(*\[Mu]/T*)\[Zeta]e,\[Zeta]\[Mu],\[Zeta]\[Tau],\[Zeta]u,\[Zeta]d,\[Zeta]c,\[Zeta]s,\[Zeta]\[Pi]c,xe,x\[Mu],x\[Tau],xu,xd,xc,xs,x\[Pi]c,(*accumulators*)\[CapitalDelta]\[Nu]e=0.,\[CapitalDelta]\[Nu]\[Mu]=0.,\[CapitalDelta]\[Nu]\[Tau]=0.,\[CapitalDelta]e=0.,\[CapitalDelta]\[Mu]=0.,\[CapitalDelta]\[Tau]=0.,\[CapitalDelta]u=0.,\[CapitalDelta]d=0.,\[CapitalDelta]c=0.,\[CapitalDelta]s=0.,\[CapitalDelta]\[Pi]=0.,\[Chi]2B,\[Chi]2Q,\[Chi]11,fps=1./(2 Pi^2),y,w,i,E\[Pi]=0.,Eu=0.,Ed=0.,Ec=0.,Es=0.},*)
(*(*bookkeeping*)*)
(*\[Zeta]e=\[Zeta]\[Nu]e-\[Zeta]Q;\[Zeta]\[Mu]=\[Zeta]\[Nu]\[Mu]-\[Zeta]Q;\[Zeta]\[Tau]=\[Zeta]\[Nu]\[Tau]-\[Zeta]Q;*)
(*\[Zeta]u=\[Zeta]B/3.+2 \[Zeta]Q/3.;\[Zeta]d=\[Zeta]B/3.-\[Zeta]Q/3.;*)
(*\[Zeta]c=\[Zeta]u;\[Zeta]s=\[Zeta]d;*)
(*\[Zeta]\[Pi]c=-\[Zeta]Q;*)
(*xe=me/T;x\[Mu]=mmu/T;x\[Tau]=mtau/T;*)
(*xu=muq/T;xd=mdq/T;xc=mcq/T;xs=msq/T;*)
(*x\[Pi]c=mpic/T;*)
(*(*Simpson integration*)*)
(*For[i=0,i<n3,i++,y=yMin3+i dy3;*)
(*w=coeSimps3[[i+1]] y^2;*)
(*(*neutrinos*)*)
(*\[CapitalDelta]\[Nu]e+=w (fFD[y,\[Zeta]\[Nu]e]-fFD[y,-\[Zeta]\[Nu]e]);*)
(*\[CapitalDelta]\[Nu]\[Mu]+=w (fFD[y,\[Zeta]\[Nu]\[Mu]]-fFD[y,-\[Zeta]\[Nu]\[Mu]]);*)
(*\[CapitalDelta]\[Nu]\[Tau]+=w (fFD[y,\[Zeta]\[Nu]\[Tau]]-fFD[y,-\[Zeta]\[Nu]\[Tau]]);*)
(*(*charged leptons*)*)
(*Module[{Ee=Sqrt[y^2+xe^2],E\[Mu]=Sqrt[y^2+x\[Mu]^2],E\[Tau]=Sqrt[y^2+x\[Tau]^2]},*)
(*\[CapitalDelta]e+=w (fFD[Ee,\[Zeta]e]-fFD[Ee,-\[Zeta]e]);*)
(*\[CapitalDelta]\[Mu]+=w (fFD[E\[Mu],\[Zeta]\[Mu]]-fFD[E\[Mu],-\[Zeta]\[Mu]]);*)
(*\[CapitalDelta]\[Tau]+=w (fFD[E\[Tau],\[Zeta]\[Tau]]-fFD[E\[Tau],-\[Zeta]\[Tau]]);];*)
(*(*quarks*)*)
(*If[T>180.,*)
(*Eu=Sqrt[y^2+xu^2];Ed=Sqrt[y^2+xd^2];Ec=Sqrt[y^2+xc^2];Es=Sqrt[y^2+xs^2];*)
(*\[CapitalDelta]u+=w (fFD[Eu,\[Zeta]u]-fFD[Eu,-\[Zeta]u]);*)
(*\[CapitalDelta]d+=w (fFD[Ed,\[Zeta]d]-fFD[Ed,-\[Zeta]d]);*)
(*\[CapitalDelta]c+=w (fFD[Ec,\[Zeta]c]-fFD[Ec,-\[Zeta]c]);*)
(*\[CapitalDelta]s+=w (fFD[Es,\[Zeta]s]-fFD[Es,-\[Zeta]s]);*)
(*];*)
(*(*pions*)*)
(*If[T<120.,*)
(*E\[Pi]=Sqrt[y^2+x\[Pi]c^2];*)
(*\[CapitalDelta]\[Pi]+=w (fBE[E\[Pi],\[Zeta]\[Pi]c]-fBE[E\[Pi],-\[Zeta]\[Pi]c])*)
(*];*)
(*];*)
(*(*prefactors*)\[CapitalDelta]\[Nu]e*=fps;\[CapitalDelta]\[Nu]\[Mu]*=fps;\[CapitalDelta]\[Nu]\[Tau]*=fps;*)
(*\[CapitalDelta]e*=2 fps;\[CapitalDelta]\[Mu]*=2 fps;\[CapitalDelta]\[Tau]*=2 fps;*)
(*\[CapitalDelta]u*=6 fps;\[CapitalDelta]d*=6 fps;\[CapitalDelta]c*=6 fps;\[CapitalDelta]s*=6 fps;*)
(*\[CapitalDelta]\[Pi]*=fps;*)
(*Module[{n=Length[tGrid],k=0},*)
(*k=idx[T,tGrid,n];(*idx is binary-search*)*)
(*\[Chi]2B=\[Chi]2Bgrid[[k]]+(T-tGrid[[k]]) s2B[[k]];*)
(*\[Chi]2Q=\[Chi]2Qgrid[[k]]+(T-tGrid[[k]]) s2Q[[k]];*)
(*\[Chi]11=\[Chi]11grid[[k]]+(T-tGrid[[k]]) s11[[k]];*)
(*];*)
(*{\[CapitalDelta]\[Nu]e,\[CapitalDelta]\[Nu]\[Mu],\[CapitalDelta]\[Nu]\[Tau],\[CapitalDelta]e,\[CapitalDelta]\[Mu],\[CapitalDelta]\[Tau],\[CapitalDelta]u,\[CapitalDelta]d,\[CapitalDelta]c,\[CapitalDelta]s,\[CapitalDelta]\[Pi],\[Chi]2B,\[Chi]2Q,\[Chi]11}]*)
(*,CompilationTarget->"C",RuntimeOptions->"Speed",Parallelization->True,CompilationOptions->{"InlineExternalDefinitions"->True,"InlineCompiledFunctions"->True}] /. ruleDown[{fFD,fBE}]/.ruleOwn[{me, mmu, mtau, muq, mdq, mcq, msq, mpic, n3, yMin3, dy3, coeSimps3,tGrid, \[Chi]2Bgrid, \[Chi]2Qgrid, \[Chi]11grid, s2B, s2Q, s11}]// ReleaseHold];*)
(*Print[Row[{checkIfCompiled[asymmetryC]}]; *)
(*(*==================================================================*)*)
(*(* SECTION C  \[Dash]  five-constraint system & Newton solver             *)*)
(*(*==================================================================*)*)
(*ClearAll[fConstraints];*)
(*(*==================================================================*)(*FIVE-constraint system f(\[Zeta]\:20d7;T)=0*)(*==================================================================*)ClearAll[fConstraints];*)
(*fConstraints[{\[Zeta]\[Nu]e_?NumericQ,\[Zeta]\[Nu]\[Mu]_?NumericQ,\[Zeta]\[Nu]\[Tau]_?NumericQ,\[Zeta]B_?NumericQ,\[Zeta]Q_?NumericQ},T_?NumericQ,{Le_,L\[Mu]_,L\[Tau]_}]:=Module[{\[Zeta]u=\[Zeta]B/3.+2 \[Zeta]Q/3.,\[Zeta]d=\[Zeta]B/3.-\[Zeta]Q/3.,\[Zeta]c,\[Zeta]s,\[Zeta]p,\[Zeta]n,\[CapitalDelta]\[Nu]e,\[CapitalDelta]\[Nu]\[Mu],\[CapitalDelta]\[Nu]\[Tau],\[CapitalDelta]e,\[CapitalDelta]\[Mu],\[CapitalDelta]\[Tau],\[CapitalDelta]u,\[CapitalDelta]d,\[CapitalDelta]c,\[CapitalDelta]s,\[CapitalDelta]\[Pi],\[Chi]2B,\[Chi]2Q,\[Chi]11,\[CapitalDelta]BQCD=0.,\[CapitalDelta]QQCD=0.,NB,NQ,\[CapitalDelta]p=0.,\[CapitalDelta]n=0.,s,k,len=Length[tGrid]},(*susceptibilities interpolation\[Dash]unchanged---------------------------*)*)
(*k=Round@findIndexC[T,tGrid,len];*)
(*If[k<1,k=1];If[k>len,k=len];*)
(*\[Chi]2B=\[Chi]2Bgrid[[k]]+(T-tGrid[[k]]) s2B[[k]];*)
(*\[Chi]2Q=\[Chi]2Qgrid[[k]]+(T-tGrid[[k]]) s2Q[[k]];*)
(*\[Chi]11=\[Chi]11grid[[k]]+(T-tGrid[[k]]) s11[[k]];*)
(*\[Zeta]c=\[Zeta]u;\[Zeta]s=\[Zeta]d;\[Zeta]p=\[Zeta]u+\[Zeta]d;\[Zeta]n=\[Zeta]B/3.;*)
(*{\[CapitalDelta]\[Nu]e,\[CapitalDelta]\[Nu]\[Mu],\[CapitalDelta]\[Nu]\[Tau],\[CapitalDelta]e,\[CapitalDelta]\[Mu],\[CapitalDelta]\[Tau],\[CapitalDelta]u,\[CapitalDelta]d,\[CapitalDelta]c,\[CapitalDelta]s,\[CapitalDelta]\[Pi],\[Chi]2B,\[Chi]2Q,\[Chi]11}=asymmetryC[T,\[Zeta]\[Nu]e,\[Zeta]\[Nu]\[Mu],\[Zeta]\[Nu]\[Tau],\[Zeta]B,\[Zeta]Q,0];*)
(*If[T>=Tcut,{\[CapitalDelta]BQCD,\[CapitalDelta]QQCD}=CorAsyQCD[T,\[Zeta]u,\[Zeta]c,\[Zeta]d,\[Zeta]s]];*)
(*If[T<120.,*)
(*Module[{xp=mp/T,xn=mn/T},*)
(*\[CapitalDelta]p=2 (mp T/(2 \[Pi]))^(3/2) Exp[-xp] 2 \[Zeta]p/T^3;*)
(*\[CapitalDelta]n=2 (mn T/(2 \[Pi]))^(3/2) Exp[-xn] 2 \[Zeta]n/T^3;*)
(*]*)
(*];*)
(*Which[(*pQCD--------------------------------------------------------------*)T>=Tcut,NB=(\[CapitalDelta]u+\[CapitalDelta]d+\[CapitalDelta]c+\[CapitalDelta]s+\[CapitalDelta]BQCD)/3.;*)
(*NQ=-(\[CapitalDelta]e+\[CapitalDelta]\[Mu]+\[CapitalDelta]\[Tau])+2/3 (\[CapitalDelta]u+\[CapitalDelta]c)-1/3 (\[CapitalDelta]d+\[CapitalDelta]s)+\[CapitalDelta]QQCD,(*\[Chi]\:2011window 120\[LessEqual]T<Tcut-----------------------------------------*)120.<=T<Tcut,NB=\[Chi]11 \[Zeta]Q+\[Chi]2B \[Zeta]B;*)
(*NQ=\[Chi]11 \[Zeta]B+\[Chi]2Q \[Zeta]Q-(\[CapitalDelta]e+\[CapitalDelta]\[Mu]+\[CapitalDelta]\[Tau]),(*hadron gas T<120\:202fMeV\[Dash]\[Dash]\[Dash]\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\:2011\[Dash]\[Dash]\[Dash]*)True,*)
(*NB=\[CapitalDelta]p+\[CapitalDelta]n;(*\[Chi]\:2011terms dropped*)NQ=-(\[CapitalDelta]e+\[CapitalDelta]\[Mu]+\[CapitalDelta]\[Tau]+\[CapitalDelta]\[Pi])+\[CapitalDelta]p;                 (*\[Chi]\:2011terms dropped*)];*)
(*(*total entropy (from patched kernel)*)s=ThermoQuantitiesC[T,\[Zeta]\[Nu]e,\[Zeta]\[Nu]\[Mu],\[Zeta]\[Nu]\[Tau],\[Zeta]B,\[Zeta]Q][[2]];*)
(*{(\[CapitalDelta]\[Nu]e+\[CapitalDelta]e)-Le s,(\[CapitalDelta]\[Nu]\[Mu]+\[CapitalDelta]\[Mu])-L\[Mu] s,(\[CapitalDelta]\[Nu]\[Tau]+\[CapitalDelta]\[Tau])-L\[Tau] s,NB-\[Eta]B s,NQ}];*)
(*(*temperature grid*)*)
(*(*tGrid=10.^Range[1,4,.03];*)*)
(*Tmin=24.;*)
(*tGrid=Table[10^x,{x,Log10[Tmin],Log10[10000.],(Log10[10000.]-Log10[Tmin])/120}]//N//Reverse;*)
(*(*the five symbolic unknowns*)*)
(*vars={\[Zeta]\[Nu]e,\[Zeta]\[Nu]\[Mu],\[Zeta]\[Nu]\[Tau],\[Zeta]B,\[Zeta]Q};*)
(*{Le,L\[Mu],L\[Tau]}={0.1,-0.1,0.};*)
(*(*walk along the grid,seeding each step with the previous root*)*)
(*tttt=Reap[FoldList[Function[{prevRoot,T},Module[{rootValues},rootValues=vars/. FindRoot[fConstraints[vars,T,{Le,L\[Mu],L\[Tau]}]==ConstantArray[0.00,5],(* <-vector*)Evaluate@Thread[{vars,prevRoot}],(* <-{var,x0}*)MaxIterations->80,AccuracyGoal->10,PrecisionGoal->10];*)
(*Sow@Join[{T},rootValues];(*store result*)rootValues                                     (*feed forward*)]],ConstantArray[0.00,5],(*initial guess at T=10 MeV*)tGrid]][[2,1]];*)
(**)
(*{\[Mu]behavior["\!\(\*SubscriptBox[\(\[Nu]\), \(e\)]\)"],\[Mu]behavior["\!\(\*SubscriptBox[\(\[Nu]\), \(\[Mu]\)]\)"],\[Mu]behavior["\!\(\*SubscriptBox[\(\[Nu]\), \(\[Tau]\)]\)"]}=tttt[[All,{1,#}]]&/@Range[2,4];*)
(*{\[Mu]behavior["e"],\[Mu]behavior["\[Mu]"],\[Mu]behavior["\[Tau]"]}=Table[{#[[1]],#[[x]]-#[[6]]}&/@tttt,{x,Range[2,4]}];*)
(*ListLogLinearPlot[{\[Mu]behavior["\!\(\*SubscriptBox[\(\[Nu]\), \(e\)]\)"],\[Mu]behavior["\!\(\*SubscriptBox[\(\[Nu]\), \(\[Mu]\)]\)"],\[Mu]behavior["\!\(\*SubscriptBox[\(\[Nu]\), \(\[Tau]\)]\)"],\[Mu]behavior["e"],\[Mu]behavior["\[Mu]"],\[Mu]behavior["\[Tau]"]},Joined->True,ImageSize->Large,Frame->True,FrameStyle->Directive[Thick,Black,20],PlotStyle->Join[{Thick,#}&/@{Darker@Red,Blue,Green},{Thick,#,Dashing[0.01]}&/@{Darker@Red,Blue,Green}],PlotLegends->Placed[Style[Row[{#}],20,Black]&/@{"\!\(\*SubscriptBox[\(\[Nu]\), \(e\)]\)","\!\(\*SubscriptBox[\(\[Nu]\), \(\[Mu]\)]\)","\!\(\*SubscriptBox[\(\[Nu]\), \(\[Tau]\)]\)","e","\[Mu]","\[Tau]"},{0.9,0.7}],PlotRange->{{10,10^4},All}]*)
(*]*)
(*]*)
(**)


(* ::Subsection:: *)
(*Final all-inclusive block*)


ThermodynamicsAndAsymmetriesTotal[Tmin_,Tmax_,Le_,L\[Mu]_,L\[Tau]_]:=Module[{\[CapitalDelta]LStrong,gsvals,tab,\[Mu]T\[Nu]es,\[Mu]T\[Nu]\[Mu]s,\[Mu]T\[Nu]\[Tau]s,\[Mu]TQs,\[Mu]Tes,\[Mu]T\[Mu]s,\[Mu]T\[Tau]s,thermo,Hvals,Tvals,thermoshifted,d\[Rho]dT,dtdTvals,tab1,\[Rho]vals,pvals,svals,MPL=mpl,MeVToGeV=10^-3.,\[CapitalDelta]=0.000001},tab=SolverAsymmetriesNoNB[Tmin,Tmax,Le,L\[Mu],L\[Tau],0.];
{Tvals,\[Mu]T\[Nu]es,\[Mu]T\[Nu]\[Mu]s,\[Mu]T\[Nu]\[Tau]s,\[Mu]TQs}=tab[[All,#]]&/@Join[Range[1,4],{-1}];
Tvals=Tvals*MeVToGeV;
{\[Mu]Tes,\[Mu]T\[Mu]s,\[Mu]T\[Tau]s}=(#-\[Mu]TQs)&/@{\[Mu]T\[Nu]es,\[Mu]T\[Nu]\[Mu]s,\[Mu]T\[Nu]\[Tau]s};
thermo=ThermoQuantitiesC[#[[1]],#[[2]],#[[3]],#[[4]],#[[5]],#[[6]]]&/@tab;
Hvals=1/(MPL*MeVToGeV) Sqrt[8 Pi/3 thermo[[All,1]]]*Tvals^2;
{\[Rho]vals,svals,pvals}=Tvals^4*thermo[[All,#]]&/@{1,2,3};
svals=svals/Tvals;
\[CapitalDelta]LStrong=thermo[[All,20]]/(svals/Tvals^3);
gsvals=svals/(2 Pi^2*Tvals^3/45);
tab1=SolverAsymmetriesNoNB[Tmin,Tmax,Le,L\[Mu],L\[Tau],\[CapitalDelta]];
{TvalsShifted,\[Mu]T\[Nu]esShifted,\[Mu]T\[Nu]\[Mu]sShifted,\[Mu]T\[Nu]\[Tau]sShifted,\[Mu]TQsShifted}=tab1[[All,#]]&/@Join[Range[1,4],{-1}];
TvalsShifted=TvalsShifted*MeVToGeV;
{\[Mu]TesShifted,\[Mu]T\[Mu]sShifted,\[Mu]T\[Tau]sShifted}=(#-\[Mu]TQsShifted)&/@{\[Mu]T\[Nu]esShifted,\[Mu]T\[Nu]\[Mu]sShifted,\[Mu]T\[Nu]\[Tau]sShifted};
thermoShifted=ThermoQuantitiesC[#[[1]],#[[2]],#[[3]],#[[4]],#[[5]],#[[6]]]&/@tab1;
\[Rho]valsShifted=TvalsShifted^4*thermoShifted[[All,1]];
d\[Rho]dTvals=(\[Rho]valsShifted-\[Rho]vals)/\[CapitalDelta];
dtdTvals=d\[Rho]dTvals/(3 Hvals*(pvals+\[Rho]vals)) MeVToGeV^-1;
{Tvals,Hvals,gsvals,\[Rho]vals,pvals,svals,\[Mu]T\[Nu]es,\[Mu]T\[Nu]\[Mu]s,\[Mu]T\[Nu]\[Tau]s,\[Mu]Tes,\[Mu]T\[Mu]s,\[Mu]T\[Tau]s,\[Mu]TQs,\[CapitalDelta]LStrong,dtdTvals}//Transpose
]


(* ::Section:: *)
(*End of package*)


(* ::Input::Initialization:: *)
End[]  (*`Private`*)

EndPackage[]
