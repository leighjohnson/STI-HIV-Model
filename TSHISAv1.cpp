/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <time.h>
#include <iostream>

using namespace std;

#include "TSHISAv1.h"
#include "StatFunctions.h"
#include "randomc.h"


int main()
{
	// int ia;
	clock_t start, finish;
	double elapsed_time;

	start = clock();
	//GenerateSample();
	RunSample();
	//Metropolis();
	//MetropolisCont(10000);
	//MaximizeLikelihood(0.0000001, "FinalSimplex.txt", "FinalSimplex2.txt");
	//MinimizeSumSquares(0.0000001, "FinalSex.txt", "FinalSex2.txt");
	//FitHIVandSexData();

	/*CurrYear = StartYear;
	ResetAll();
	SetCalibParameters();
	cout<<"Error ind: "<<ErrorInd<<endl;

	int iy;
	for(iy=0; iy<ProjectionTerm; iy++){
		OneYearProj();
		if(KeepCurrYearFixed==0){
			CurrYear += 1;}
		if(MaleHigh.S11.NumbersByHIVstage[5][1]>0.0){ //arbitrary check
			continue;}
		else{
			cout<<"Break in year "<<CurrYear-1<<endl;
			break;}
	}
	CalcTotalLogL();
	if(SexCalib==1){CalcSumSquares();}
	if(StoreHIVcalib==1){
		SaveHIVcalibOutput("HIVcalibOutput.txt");
		SaveTrend(HIVprevFSW, "HIVprevFSW.txt");
		SaveTrend(HIVprev15to49M, "HIVprev15to49M.txt");
		SaveTrend(HIVprev15to49F, "HIVprev15to49F.txt");
	}
	if(StoreSexCalib==1){
		SaveSexCalibOutput("SexCalibOutput.txt");
		SaveTrend(PropnActsProtectedM, "CondomUseM.txt");
		SaveTrend(PropnActsProtectedF, "CondomUseF.txt");
	}
	if(NGcalib==1){
		for(iy=0; iy<NGtransitionF.ANClogL.Observations; iy++){
			NGprevANC[iy] = NGtransitionF.ANClogL.ModelPrev[iy];}
		for(iy=0; iy<NGtransitionF.FPClogL.Observations; iy++){
			NGprevFPC[iy] = NGtransitionF.FPClogL.ModelPrev[iy];}
		SaveTrend(NGprevFSW, "NGprevFSW.txt");
		SaveTrend(NGprevANC, "NGprevANC.txt");
		SaveTrend(NGprevFPC, "NGprevFPC.txt");
		SaveTrend(NGprev15to49M, "NGprev15to49M.txt");
		SaveTrend(NGprev15to49F, "NGprev15to49F.txt");
	}*/
	//RecordAllSTDpropns();

	//TotalLogL += SumSquares;
	//cout<<"TotalLogL: "<<TotalLogL<<endl;
	/*cout<<"M HouseholdLogL.LogL: "<<NGtransitionM.HouseholdLogL.LogL<<endl;
	cout<<"F HouseholdLogL.LogL: "<<NGtransitionF.HouseholdLogL.LogL<<endl;
	cout<<"ANClogL.LogL: "<<NGtransitionF.ANClogL.LogL<<endl;
	cout<<"FPClogL.LogL: "<<NGtransitionF.FPClogL.LogL<<endl;
	cout<<"CSWlogL.LogL: "<<NGtransitionF.CSWlogL.LogL<<endl;*/
	//cout<<"ErrorCount: "<<ErrorCount<<endl;

	/*cout<<"TotalPopSum[3][0]: "<<TotalPopSum[3][0]<<endl;
	cout<<"VirginsSum[0][0]: "<<VirginsSum[0][0]<<endl;
	cout<<"MarriedSum[14][0]: "<<MarriedSum[14][0]<<endl;
	cout<<"MultPartnerSum[4][0]: "<<MultPartnerSum[4][0]<<endl;
	cout<<"MultPartnerSum[4][1]: "<<MultPartnerSum[4][1]<<endl;
	cout<<"LowRiskSum[1][0]: "<<LowRiskSum[1][0]<<endl;
	cout<<"LowRiskSum[1][1]: "<<LowRiskSum[1][1]<<endl;
	cout<<"HIVstageSumM[5][0]: "<<HIVstageSumM[5][0]<<endl;
	cout<<"HIVstageSumM[5][1]: "<<HIVstageSumM[5][1]<<endl;
	cout<<"HIVstageSumM[5][2]: "<<HIVstageSumM[5][2]<<endl;
	cout<<"HIVstageSumF[5][0]: "<<HIVstageSumF[5][0]<<endl;
	cout<<"HIVstageSumF[5][1]: "<<HIVstageSumF[5][1]<<endl;
	cout<<"HIVstageSumF[5][2]: "<<HIVstageSumF[5][2]<<endl;
	cout<<"TotalBirths: "<<TotalBirths<<endl;
	cout<<"NewHIVbreastmilk: "<<NewHIVbreastmilk<<endl;
	cout<<"MaleChild.TotalAlive[2]: "<<MaleChild.TotalAlive[2]<<endl;
	cout<<"FemChild.Perinatal.PreAIDSstage[1]: "<<FemChild.Perinatal.PreAIDSstage[1]<<endl;
	cout<<"MaleHigh.L1.NumbersByHIVstage[5][0]: "<<MaleHigh.L1.NumbersByHIVstage[5][0]<<endl;
	cout<<"MaleHigh.L1.NumbersByHIVstage[5][1]: "<<MaleHigh.L1.NumbersByHIVstage[5][1]<<endl;
	cout<<"MaleHigh.L1.NumbersByHIVstage[5][2]: "<<MaleHigh.L1.NumbersByHIVstage[5][2]<<endl;*/

	/*cout<<"TPtransitionM.AliveStage0[3][0]: "<<TPtransitionM.AliveStage0[3][0]<<endl;
	cout<<"HSVtransitionM.AliveStage0[2][1]: "<<HSVtransitionM.AliveStage0[2][1]<<endl;
	cout<<"HDtransitionM.AliveStage1[4][1]: "<<HDtransitionM.AliveStage1[4][1]<<endl;
	cout<<"CTtransitionM.AliveStage2[1][2]: "<<CTtransitionM.AliveStage2[1][2]<<endl;
	cout<<"NGtransitionF.AliveStage0[3][1]: "<<NGtransitionF.AliveStage0[3][1]<<endl;
	cout<<"TVtransitionF.AliveStage1[5][2]: "<<TVtransitionF.AliveStage1[5][2]<<endl;
	cout<<"BVtransitionF.AliveStage2[2][0]: "<<BVtransitionF.AliveStage2[2][0]<<endl;
	cout<<"VCtransitionF.AliveStage0[6][1]: "<<VCtransitionF.AliveStage0[6][1]<<endl;*/

	//cout<<"SumSquares: "<<SumSquares<<endl;
	/*cout<<"ANC 1991 20: "<<HIVtransitionF.AntenatalNlogL.ModelPrev[1]<<endl;
	cout<<"ANC 1996 20: "<<HIVtransitionF.AntenatalNlogL.ModelPrev[26]<<endl;
	cout<<"ANC 2001 20: "<<HIVtransitionF.AntenatalNlogL.ModelPrev[51]<<endl;
	cout<<"ANC 2006 20: "<<HIVtransitionF.AntenatalNlogL.ModelPrev[76]<<endl;
	cout<<"ANC 2006 15: "<<HIVtransitionF.AntenatalNlogL.ModelPrev[75]<<endl;
	cout<<"ANC 2006 25: "<<HIVtransitionF.AntenatalNlogL.ModelPrev[77]<<endl;
	cout<<"ANC 2006 30: "<<HIVtransitionF.AntenatalNlogL.ModelPrev[78]<<endl;
	cout<<"ANC 2006 35: "<<HIVtransitionF.AntenatalNlogL.ModelPrev[79]<<endl;
	cout<<"HH Male 2005 20: "<<HIVtransitionM.HouseholdNlogL.ModelPrev[11]<<endl;
	cout<<"HH Male 2005 30: "<<HIVtransitionM.HouseholdNlogL.ModelPrev[13]<<endl;
	cout<<"HH Male 2005 40: "<<HIVtransitionM.HouseholdNlogL.ModelPrev[15]<<endl;
	cout<<"HH Male 2005 50: "<<HIVtransitionM.HouseholdNlogL.ModelPrev[17]<<endl;*/

	finish = clock();
	elapsed_time = (finish - start);
	cout<<"Time taken: "<<elapsed_time/1000.0<<" seconds"<<endl;
	return 0;
}

PrevalenceData::PrevalenceData(){}

NationalData::NationalData()
{
	int ia;

	for(ia=0; ia<16; ia++){
		BiasMult[ia] = 1.0;}
}

void NationalData::CalcLogL()
{
	int ia, ic;
	double AdjPrev, ModelVarEst;
	double VarLogitPrev[100]; // Variance of the logit-transformed prevalence estimates 
							  // (sampling variation only). Change dimension if n>100.
	double SpSum, BiasSum, BiasLevel=0.0, BiasTrend;

	BiasSum = 0.0;
	for(ia=0; ia<16; ia++){
		BiasSum += log(BiasMult[ia]);}
	SpSum = 0.0;
	for(ic=0; ic<Observations; ic++){
		SpSum += 1.0 - ExpSp[ic];}

	if(BiasSum > 0.0 || SpSum > 0.0){
		BiasLevel = 0.0;
		for(ic=0; ic<Observations; ic++){
			BiasLevel += log(StudyPrev[ic]/(1.0 - StudyPrev[ic])) - log(ModelPrev[ic]/(1.0 - 
				ModelPrev[ic]));}
		BiasLevel = BiasLevel/Observations;
		if(BiasLevel < 0.0){BiasLevel = 0.0;}
		if(FixedUncertainty==1){OutANCbias.out[CurrSim-1][0] = exp(BiasLevel);}
		// The following 6 lines of code have been added to test for trend in the bias
		/*BiasTrend = 0.0;
		for(ic=0; ic<Observations; ic++){
			BiasTrend += (log(StudyPrev[ic]/(1.0 - StudyPrev[ic])) - log(ModelPrev[ic]/(1.0 - 
				ModelPrev[ic])) - BiasLevel) * (StudyYear[ic] - 2001.0)/300.0;
			//BiasTrend += (log(StudyPrev[ic]/(1.0 - StudyPrev[ic])) - log(ModelPrev[ic]/(1.0 - 
			//	ModelPrev[ic])) - BiasLevel) * (AgeStart[ic] - 3.0)/10.0;
		}
		if(FixedUncertainty==1){OutANCbiasTrend.out[CurrSim-1][0] = BiasTrend;}*/
	}

	LogL = 0;
	ModelVarEst = 0;
	for(ic=0; ic<Observations; ic++){
		AdjPrev = 1.0/(1.0 + (1.0/ModelPrev[ic] - 1.0) * exp(-BiasLevel));
		//AdjPrev = 1.0/(1.0 + (1.0/ModelPrev[ic] - 1.0) * exp(-BiasLevel - BiasTrend * 
		//	(StudyYear[ic] - 2001.0)));
		//AdjPrev = 1.0/(1.0 + (1.0/ModelPrev[ic] - 1.0) * exp(-BiasLevel - BiasTrend * 
		//	(AgeStart[ic] - 3.0)));
		if(GetSDfromData==0){
			LogL += -0.5 * log(2.0 * 3.141592654) - log(PrevSE[ic])
				- 0.5 * pow((StudyPrev[ic] - AdjPrev)/PrevSE[ic], 2.0);}
		else{
			VarLogitPrev[ic] = pow(PrevSE[ic]/(StudyPrev[ic] * (1.0 - StudyPrev[ic])), 2.0);
			ModelVarEst += pow(log(StudyPrev[ic]/(1.0 - StudyPrev[ic])) - log(AdjPrev/(1.0 - 
				AdjPrev)), 2.0) - VarLogitPrev[ic];
		}
		ModelPrev[ic] = AdjPrev;
	}

	if(GetSDfromData==1){
		if(ModelVarEst < 0.0 || BiasSum==0.0){
			ModelVarEst = 0;}
		else{
			ModelVarEst = ModelVarEst/Observations;}
		if(FixedUncertainty==1){
			if(BiasSum > 0.0 || SpSum > 0.0){
				OutModelVarANC.out[CurrSim-1][0] = ModelVarEst;}
			else if(SexInd==1){
				OutModelVarHH.out[CurrSim-1][0] = ModelVarEst;}
			else{
				OutModelVarHH.out[CurrSim-1][1] = ModelVarEst;}
		}
		for(ic=0; ic<Observations; ic++){
			LogL += -0.5 * (log(2.0 * 3.141592654 * (VarLogitPrev[ic] + ModelVarEst)) +
				pow(log(StudyPrev[ic]/(1.0 - StudyPrev[ic])) - log(ModelPrev[ic]/(1.0 - 
				ModelPrev[ic])), 2.0)/(VarLogitPrev[ic] + ModelVarEst));
		}
	}
}

AntenatalN::AntenatalN(){}

HouseholdN::HouseholdN(){}

SentinelData::SentinelData()
{
	BiasMult = 1.0;
}

void SentinelData::CalcLogL()
{
	int ic;
	double Mean, Var; // The mean and variance of theta(i), the modelled prevalence
					  // of the STD in study i
	double alpha, beta; // The alpha and beta parameters for the beta prior on theta(i)
	double a, b, c, d; // Arguments for the gamma functions
	double LogLikelihood;

	LogL = 0;
	for(ic=0; ic<Observations; ic++){
		if(ModelPrev[ic]<0.0){ModelPrev[ic] = 0.000000001;} // Negative values are possible due to rounding
		ModelPrev[ic] *= BiasMult;
		Mean = 1.0 - ExpSp[ic] + ModelPrev[ic] * (ExpSe[ic] + ExpSp[ic] - 1.0) +
			0.5 * VarStudyEffect * (ExpSe[ic] + ExpSp[ic] - 1.0) * ModelPrev[ic] * (1.0 - 
			ModelPrev[ic]) * (1.0 - 2.0 * ModelPrev[ic]);
		Var = VarSe[ic] * pow(ModelPrev[ic], 2.0) * (1.0 + VarStudyEffect * pow(1.0 - 
			ModelPrev[ic], 2.0)) + VarSp[ic] * pow(1.0 - ModelPrev[ic], 2.0) * (1.0 +
			VarStudyEffect * pow(ModelPrev[ic], 2.0)) + 0.5 * pow(VarStudyEffect, 2.0) *
			pow((ExpSe[ic] + ExpSp[ic] - 1.0) * ModelPrev[ic] * (1.0 - ModelPrev[ic]) * 
			(1.0 - 2.0 * ModelPrev[ic]), 2.0) + VarStudyEffect * pow((ExpSe[ic] + ExpSp[ic] - 
			1.0) * ModelPrev[ic] * (1.0 - ModelPrev[ic]), 2.0);
		StudyPos[ic] = StudyN[ic] * StudyPrev[ic];

		if(Var>0){
			alpha = Mean * (Mean * (1.0 - Mean)/Var - 1.0);
			beta = (1.0 - Mean) * (Mean * (1.0 - Mean)/Var - 1.0);
	
			a = alpha + beta;
			b = alpha + StudyPos[ic];
			c = beta + StudyN[ic] - StudyPos[ic];
			d = alpha + beta + StudyN[ic];

			LogLikelihood = gamma_log(&a) + gamma_log(&b) + gamma_log(&c) - gamma_log(&d) -
				gamma_log(&alpha) - gamma_log(&beta);
		}
		else{
			// In this case, the mean is fixed, so the likelihood is just the binomial.
			a = StudyN[ic] + 1.0;
			b = StudyPos[ic] + 1.0;
			c = StudyN[ic] - StudyPos[ic] + 1.0;
			LogLikelihood = gamma_log(&a) - gamma_log(&b) - gamma_log(&c) + StudyPos[ic] *
				log(Mean) + (StudyN[ic] - StudyPos[ic]) * log(1.0 - Mean);
		}
		LogL += LogLikelihood;
	}
}

Household::Household(){}

NonHousehold::NonHousehold(){}

ANC::ANC(){}

FPC::FPC(){}

GUD::GUD(){}

CSW::CSW(){}

OutputArray::OutputArray(int n)
{
	columns = n;
}

void OutputArray::Record(const char* filout, int n)
{
	int i, c;
	ofstream file(filout);

	for(i=0; i<simulations; i++){
		file<<setw(6)<<right<<i;
		for(c=0; c<columns; c++){
			file<<"	"<<setw(10)<<right<<out[i][c];}
		file<<endl;
	}
	file.close();
}

void OutputArray::RecordSample(const char* filout, int n)
{
	int i, c;
	ofstream file(filout);

	for(i=0; i<samplesize; i++){
		file<<setw(6)<<right<<i<<"	"<<setw(6)<<right<<sampleid[i];
		for(c=0; c<columns; c++){
			file<<"	"<<setw(10)<<right<<out[i][c];}
		file<<endl;
	}
	file.close();
}

void OutputArray::SampleInput()
{
	int i, c;
	double temp[samplesize][41];
	
	for(i=0; i<samplesize; i++){
		for(c=0; c<columns; c++){
			temp[i][c] = out[sampleid[i]][c];}
	}
	for(i=0; i<samplesize; i++){
		for(c=0; c<columns; c++){
			out[i][c] = temp[i][c];}
	}
}

double MCMCparameter::GetLogPrior(double x)
{
	double y, LogPrior;

	if(PriorType==1 || PriorType==3){
		y = 1.0/(1.0 + exp(-x));} // Reverse logit transformation for beta and uniform
	else{
		y = exp(x);} // Reverse log transformation for gamma

	if(PriorType==1){
		LogPrior = (Param1 - 1.0) * log(y) + (Param2 - 1.0) * log(1.0 - y);}
	if(PriorType==2){
		LogPrior = Param1 * log(Param2) + (Param1 - 1.0) * log(y) - Param2 * y;}
	if(PriorType==3){
		if(y<Param1 || y>Param2){
			LogPrior = -1000.0;} // Arbitrary value, sufficiently low to prevent sampling
		else{
			LogPrior = log(1.0/(Param2 - Param1));}
	}

	// Now add the log of the Jacobian (note that we work with x, not y)
	if(PriorType==1 || PriorType==3){
		LogPrior = LogPrior - x - 2.0 * log(1.0 + exp(-x));}
	else{
		LogPrior += x;}

	return LogPrior;
}

void MCMCparameter::GetMean(int n)
{
	int i, j, start;

	start = CumIterations - n;
	if(start<0){
		start = 0;}

	mean = 0.0;
	for(i=start; i<CumIterations; i++){
		for(j=0; j<columns; j++){
			mean += out[i][j];}
	}
	mean = mean/(columns * (CumIterations - start));
}

void MCMCparameter::GetCovar(MCMCparameter* a, int n)
{
	int i, j, start;
	double crossprod;

	start = CumIterations - n;
	if(start<0){
		start = 0;}

	if(index >= a->index){
		crossprod = 0;
		for(i=start; i<CumIterations; i++){
			for(j=0; j<columns; j++){
				crossprod += (out[i][j] - mean) * (a->out[i][j] - a->mean);}
		}
		Covariance[index][a->index] = crossprod/(columns * (CumIterations - start) - 1.0);
		if(index != a->index){
			Covariance[a->index][index] = Covariance[index][a->index];}
	}
}

void MCMCparameter::GetAllCovar(int n)
{
	//GetCovar(&lMtoFtransmCSW, n);
	GetCovar(&lMtoFtransmST, n);
	GetCovar(&lMtoFtransmLT, n);
	//GetCovar(&lFtoMtransmCSW, n);
	GetCovar(&lFtoMtransmST, n);
	GetCovar(&lFtoMtransmLT, n);
	GetCovar(&lInitHIVprevHigh, n);
	//GetCovar(&lHighPropnM, n);
	//GetCovar(&lHighPropnF, n);
	GetCovar(&lRelPartnerAcqM, n);
	GetCovar(&lRelPartnerAcqF, n);
	GetCovar(&lAssortativeness, n);
	//GetCovar(&lMeanPartnerDur, n);
	//GetCovar(&lMeanPartnerGapM, n);
	//GetCovar(&lMeanPartnerGapF, n);
	GetCovar(&lRelPartnerAcqMM, n);
	GetCovar(&lRelPartnerAcqMF, n);
	GetCovar(&lRelPartnerAcqLM, n);
	GetCovar(&lRelPartnerAcqLF, n);
	GetCovar(&lCondomBias, n);
	GetCovar(&lRelARTinfectiousness, n);
}

void MCMCparameter::TestConvergence(double pi)
{
	int i, j, start;

	start = (1.0 - pi) * CumIterations;
	mean = 0.0;
	for(j=0; j<columns; j++){
		SeriesMean[j] = 0.0;
		for(i=start; i<CumIterations; i++){
			SeriesMean[j] += out[i][j];}
		SeriesMean[j] = SeriesMean[j]/(CumIterations - start);
		mean += SeriesMean[j];
		SeriesVar[j] = 0.0;
		for(i=start; i<CumIterations; i++){
			SeriesVar[j] += pow(out[i][j] - SeriesMean[j], 2.0);}
		SeriesVar[j] = SeriesVar[j]/(CumIterations - start - 1.0);
	}
	mean = mean/columns;

	BetweenVar = 0.0;
	WithinVar = 0.0;
	for(j=0; j<columns; j++){
		BetweenVar += pow(SeriesMean[j] - mean, 2.0);
		WithinVar += SeriesVar[j];
	}
	BetweenVar *= (CumIterations - start)/(columns - 1.0);
	WithinVar = WithinVar/columns;

	ScaleRedn = pow(1.0 + (BetweenVar/WithinVar - 1.0)/(CumIterations - start), 0.5);
	cout.precision(5);
	cout<<"ScaleRedn for parameter "<<index<<": "<<ScaleRedn<<endl;
}

void MCMCparameter::ReadCumIterations(const char* input)
{
	int i, j, idum;
	ifstream file;

	file.open(input);
	if(file==0)cout<<"File open error"<<endl;
	for(i=0; i<CumIterations; i++){
		file>>idum;
		for(j=0; j<NumberSeries; j++){
			file>>out[i][j];}
	}
	file.close();
}

PostOutputArray::PostOutputArray(int n)
{
	columns = n;
}

void PostOutputArray::RecordSample(const char* filout)
{
	int i, c;
	ofstream file(filout);

	for(i=0; i<samplesize; i++){
		file<<setw(6)<<right<<i<<"	"<<setw(6)<<right<<sampleid[i];
		for(c=0; c<columns; c++){
			file<<"	"<<setw(10)<<right<<out[i][c];}
		file<<endl;
	}
	file.close();
}

STDtransition::STDtransition(){}

void STDtransition::ClearTransmProb()
{
	int ia;

	for(ia=0; ia<16; ia++){
		TransmProbS1to1[ia] = 0;
		TransmProbS1to2[ia] = 0;
		TransmProbS2to1[ia] = 0;
		TransmProbS2to2[ia] = 0;
		TransmProbL1to1[ia] = 0;
		TransmProbL1to2[ia] = 0;
		TransmProbL2to1[ia] = 0;
		TransmProbL2to2[ia] = 0;
	}
}

void STDtransition::CalcTransmProb()
{
	int ia;
	double denominator;

	for(ia=0; ia<16; ia++){
		if(SexInd==0){
			denominator = MaleHigh.S1.TotalAlive[ia] + 
				2.0 * MaleHigh.S11.TotalAlive[ia] + MaleHigh.S12.TotalAlive[ia] +
				MaleHigh.L11.TotalAlive[ia] + MaleHigh.L21.TotalAlive[ia];
			if(denominator>0){TransmProbS1to1[ia] = TransmProbS1to1[ia]/denominator;}
			denominator = MaleHigh.S2.TotalAlive[ia] +
				MaleHigh.S12.TotalAlive[ia] + 2.0 * MaleHigh.S22.TotalAlive[ia] +
				MaleHigh.L12.TotalAlive[ia] + MaleHigh.L22.TotalAlive[ia];
			if(denominator>0){TransmProbS1to2[ia] = TransmProbS1to2[ia]/denominator;}
			if(MaleLow.S1.TotalAlive[ia]>0){
				TransmProbS2to1[ia] = TransmProbS2to1[ia]/MaleLow.S1.TotalAlive[ia];}
			if(MaleLow.S2.TotalAlive[ia]>0){
				TransmProbS2to2[ia] = TransmProbS2to2[ia]/MaleLow.S2.TotalAlive[ia];}
			denominator = MaleHigh.L1.TotalAlive[ia] +
				MaleHigh.L11.TotalAlive[ia] + MaleHigh.L12.TotalAlive[ia];
			if(denominator>0){TransmProbL1to1[ia] = TransmProbL1to1[ia]/denominator;}
			denominator = MaleHigh.L2.TotalAlive[ia] +
				MaleHigh.L21.TotalAlive[ia] + MaleHigh.L22.TotalAlive[ia];
			if(denominator>0){TransmProbL1to2[ia] = TransmProbL1to2[ia]/denominator;}
			if(MaleLow.L1.TotalAlive[ia]>0){
				TransmProbL2to1[ia] = TransmProbL2to1[ia]/MaleLow.L1.TotalAlive[ia];}
			if(MaleLow.L2.TotalAlive[ia]>0){
				TransmProbL2to2[ia] = TransmProbL2to2[ia]/MaleLow.L2.TotalAlive[ia];}
		}
		else{
			denominator = FemHigh.S1.TotalAlive[ia] + 
				2.0 * FemHigh.S11.TotalAlive[ia] + FemHigh.S12.TotalAlive[ia] +
				FemHigh.L11.TotalAlive[ia] + FemHigh.L21.TotalAlive[ia];
			if(denominator>0){TransmProbS1to1[ia] = TransmProbS1to1[ia]/denominator;}
			denominator = FemHigh.S2.TotalAlive[ia] +
				FemHigh.S12.TotalAlive[ia] + 2.0 * FemHigh.S22.TotalAlive[ia] +
				FemHigh.L12.TotalAlive[ia] + FemHigh.L22.TotalAlive[ia];
			if(denominator>0){TransmProbS1to2[ia] = TransmProbS1to2[ia]/denominator;}
			if(FemLow.S1.TotalAlive[ia]>0){
				TransmProbS2to1[ia] = TransmProbS2to1[ia]/FemLow.S1.TotalAlive[ia];}
			if(FemLow.S2.TotalAlive[ia]>0){
				TransmProbS2to2[ia] = TransmProbS2to2[ia]/FemLow.S2.TotalAlive[ia];}
			denominator = FemHigh.L1.TotalAlive[ia] +
				FemHigh.L11.TotalAlive[ia] + FemHigh.L12.TotalAlive[ia];
			if(denominator>0){TransmProbL1to1[ia] = TransmProbL1to1[ia]/denominator;}
			denominator = FemHigh.L2.TotalAlive[ia] +
				FemHigh.L21.TotalAlive[ia] + FemHigh.L22.TotalAlive[ia];
			if(denominator>0){TransmProbL1to2[ia] = TransmProbL1to2[ia]/denominator;}
			if(FemLow.L1.TotalAlive[ia]>0){
				TransmProbL2to1[ia] = TransmProbL2to1[ia]/FemLow.L1.TotalAlive[ia];}
			if(FemLow.L2.TotalAlive[ia]>0){
				TransmProbL2to2[ia] = TransmProbL2to2[ia]/FemLow.L2.TotalAlive[ia];}
		}
	}
}

void STDtransition::CalcInfectProb(double Transm[16], double Infect[16], int RelType)
{
	int ia, ib;

	// NB: Remember that Infect[16], CondomEff and SuscepIncrease[16] all belong to the
	// opposite sex, which is why we use the sex opposite to that indicated by SexInd in
	// this function.

	for(ia=0; ia<16; ia++){
		Infect[ia] = 0;
		if(SexInd==0){
			for(ib=0; ib<16; ib++){Infect[ia] += Transm[ib] * AgePrefF[ia][ib];}
		}
		else{
			for(ib=0; ib<16; ib++){Infect[ia] += Transm[ib] * AgePrefM[ia][ib];}
		}
		Infect[ia] *= SuscepIncrease[ia];
		if(RelType==0){
			Infect[ia] *= (1.0 - CondomUseST[ia][1-SexInd] * CondomEff);}
		else{
			Infect[ia] *= (1.0 - CondomUseLT[ia][1-SexInd] * CondomEff);}
		if(Infect[ia]>1){
			Infect[ia] = 1.0;}
	}
}

void STDtransition::CalcAllInfectProb()
{
	CalcInfectProb(TransmProbS1to1, InfectProbS1from1, 0);
	CalcInfectProb(TransmProbS1to2, InfectProbS2from1, 0);
	CalcInfectProb(TransmProbS2to1, InfectProbS1from2, 0);
	CalcInfectProb(TransmProbS2to2, InfectProbS2from2, 0);
	CalcInfectProb(TransmProbL1to1, InfectProbL1from1, 1);
	CalcInfectProb(TransmProbL1to2, InfectProbL2from1, 1);
	CalcInfectProb(TransmProbL2to1, InfectProbL1from2, 1);
	CalcInfectProb(TransmProbL2to2, InfectProbL2from2, 1);

	// Calculate infection prob for interactions between FSWs and clients
	
	int ia;
	double BaseInfectProb, TotalFSW;

	if(SexInd==0){
		BaseInfectProb = InfectProbFSW[0] * (1.0 - CondomUseFSW * CondomEff)/
			(MaleHigh.S1.AnnFSWcontacts + MaleHigh.S2.AnnFSWcontacts + 
			MaleHigh.L1.AnnFSWcontacts + MaleHigh.L2.AnnFSWcontacts +
			MaleHigh.S11.AnnFSWcontacts + MaleHigh.S12.AnnFSWcontacts +
			MaleHigh.S22.AnnFSWcontacts + MaleHigh.L11.AnnFSWcontacts +
			MaleHigh.L12.AnnFSWcontacts + MaleHigh.L21.AnnFSWcontacts +
			MaleHigh.L22.AnnFSWcontacts + MaleHigh.NoPartner.AnnFSWcontacts);
		if(RelTransmCSW > 0){
			BaseInfectProb *= RelTransmCSW;}
	}
	else{
		TotalFSW = 0;
		for(ia=0; ia<16; ia++){
			TotalFSW += FemHigh.FSW.TotalAlive[ia];}
		BaseInfectProb = InfectProbFSW[0] * (1.0 - CondomUseFSW * CondomEff)/TotalFSW;
	}
	for(ia=0; ia<16; ia++){
		InfectProbFSW[ia] = BaseInfectProb * SuscepIncrease[ia];}
}

void STDtransition::ReadPrevData(const char* input)
{
	int ic;
	ifstream file;

	file.open(input);
	if(file==0)cout<<"File open error"<<endl;
	file.ignore(255,'\n');
	if(ANClogL.Observations>0){
		for(ic=0; ic<ANClogL.Observations; ic++){
			file>>ANClogL.StudyYear[ic]>>ANClogL.StudyN[ic]>>ANClogL.StudyPrev[ic]>>
				ANClogL.ExpSe[ic]>>ANClogL.ExpSp[ic]>>ANClogL.VarSe[ic]>>ANClogL.VarSp[ic]>>
				ANClogL.HIVprevInd[ic]>>ANClogL.HIVprev[ic];}
		file.ignore(255,'\n');
	}
	file.ignore(255,'\n');
	if(FPClogL.Observations>0){
		for(ic=0; ic<FPClogL.Observations; ic++){
			file>>FPClogL.StudyYear[ic]>>FPClogL.StudyN[ic]>>FPClogL.StudyPrev[ic]>>
				FPClogL.ExpSe[ic]>>FPClogL.ExpSp[ic]>>FPClogL.VarSe[ic]>>FPClogL.VarSp[ic]>>
				FPClogL.HIVprevInd[ic]>>FPClogL.HIVprev[ic];}
		file.ignore(255,'\n');
	}
	file.ignore(255,'\n');
	if(GUDlogL.Observations>0){
		for(ic=0; ic<GUDlogL.Observations; ic++){
			file>>GUDlogL.StudyYear[ic]>>GUDlogL.StudyN[ic]>>GUDlogL.StudyPrev[ic]>>
				GUDlogL.ExpSe[ic]>>GUDlogL.ExpSp[ic]>>GUDlogL.VarSe[ic]>>GUDlogL.VarSp[ic]>>
				GUDlogL.HIVprevInd[ic]>>GUDlogL.HIVprev[ic];}
		file.ignore(255,'\n');
	}
	file.ignore(255,'\n');
	if(CSWlogL.Observations>0){
		for(ic=0; ic<CSWlogL.Observations; ic++){
			file>>CSWlogL.StudyYear[ic]>>CSWlogL.StudyN[ic]>>CSWlogL.StudyPrev[ic]>>
				CSWlogL.ExpSe[ic]>>CSWlogL.ExpSp[ic]>>CSWlogL.VarSe[ic]>>CSWlogL.VarSp[ic]>>
				CSWlogL.HIVprevInd[ic]>>CSWlogL.HIVprev[ic];}
		file.ignore(255,'\n');
	}
	file.ignore(255,'\n');
	if(HouseholdLogL.Observations>0){
		for(ic=0; ic<HouseholdLogL.Observations; ic++){
			file>>HouseholdLogL.StudyYear[ic]>>HouseholdLogL.StudyN[ic]>>
				HouseholdLogL.StudyPrev[ic]>>HouseholdLogL.ExpSe[ic]>>
				HouseholdLogL.ExpSp[ic]>>HouseholdLogL.VarSe[ic]>>HouseholdLogL.VarSp[ic]>>
				HouseholdLogL.HIVprevInd[ic]>>HouseholdLogL.HIVprev[ic]>>
				HouseholdLogL.AgeStart[ic]>>HouseholdLogL.AgeEnd[ic];}
		file.ignore(255,'\n');
	}
	file.ignore(255,'\n');
	if(AntenatalNlogL.Observations>0){
		for(ic=0; ic<AntenatalNlogL.Observations; ic++){
			file>>AntenatalNlogL.StudyYear[ic]>>AntenatalNlogL.AgeStart[ic]>>
				AntenatalNlogL.StudyPrev[ic]>>AntenatalNlogL.PrevSE[ic]>>
				AntenatalNlogL.ExpSe[ic]>>AntenatalNlogL.ExpSp[ic];}
		file.ignore(255,'\n');
	}
	file.ignore(255,'\n');
	if(HouseholdNlogL.Observations>0){
		for(ic=0; ic<HouseholdNlogL.Observations; ic++){
			file>>HouseholdNlogL.StudyYear[ic]>>HouseholdNlogL.AgeStart[ic]>>
				HouseholdNlogL.StudyPrev[ic]>>HouseholdNlogL.PrevSE[ic]>>
				HouseholdNlogL.ExpSe[ic]>>HouseholdNlogL.ExpSp[ic];}
	}
	file.close();
}

void STDtransition::GetCSWprev()
{
	int ic;

	if(CSWlogL.Observations>0){
		for(ic=0; ic<CSWlogL.Observations; ic++){
			if(CurrYear==CSWlogL.StudyYear[ic]){
				CSWlogL.ModelPrev[ic] = CSWprevalence;}
		}
	}
}

void STDtransition::SetVarStudyEffect(double Variance)
{
	ANClogL.VarStudyEffect = Variance;
	FPClogL.VarStudyEffect = Variance;
	CSWlogL.VarStudyEffect = Variance;
	GUDlogL.VarStudyEffect = Variance;
	HouseholdLogL.VarStudyEffect = Variance;
}

HIVtransition::HIVtransition(int Sex, int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, 
							 int ObsHH, int ObsANCN, int ObsHHN)
{
	SexInd = Sex;
	nStates = 6;

	ANClogL.Observations = ObsANC;
	FPClogL.Observations = ObsFPC;
	GUDlogL.Observations = ObsGUD;
	CSWlogL.Observations = ObsCSW;
	HouseholdLogL.Observations = ObsHH;
	AntenatalNlogL.Observations = ObsANCN;
	HouseholdNlogL.Observations = ObsHHN;
	HouseholdNlogL.SexInd = Sex;
}

void HIVtransition::CalcTransitionProbs()
{
	int iy, i;
	double TransmProbAdj;

	iy = CurrYear - StartYear;
	From1to2 = 1.0 - exp(-(1.0/AveDuration[0]) * 52.0/CycleD);
	From2to3 = 1.0 - exp(-(1.0/AveDuration[1]) * 52.0/CycleD);
	From3to4 = (1.0 - exp(-(1.0/AveDuration[2]) * 52.0/CycleD)) * (1.0 - HAARTaccess[iy]);
	From3to5 = (1.0 - exp(-(1.0/AveDuration[2]) * 52.0/CycleD)) * HAARTaccess[iy];
	From4toDead = (1.0 - exp(-(1.0/AveDuration[3]) * 52.0/CycleD));
	From5toDead = (1.0 - exp(-(1.0/AveDuration[4]) * 52.0/CycleD));
}

void HIVtransition::GetANCprev()
{
	int ic;

	if(AntenatalNlogL.Observations>0){
		for(ic=0; ic<AntenatalNlogL.Observations; ic++){
			if(AntenatalNlogL.StudyYear[ic]==CurrYear){
				AntenatalNlogL.ModelPrev[ic] = BirthsToHIVmothers[AntenatalNlogL.
					AgeStart[ic]-1]/BirthsByAge[AntenatalNlogL.AgeStart[ic]-1];}
		}
	}

	if(ANClogL.Observations>0){
		for(ic=0; ic<ANClogL.Observations; ic++){
			if(ANClogL.StudyYear[ic]==CurrYear){
				ANClogL.ModelPrev[ic] = TotBirthsToHIVmothers/TotalBirths;}
		}
	}
}

void HIVtransition::GetHHprev()
{
	int ic, ia;
	double numerator, denominator;

	if(HouseholdNlogL.Observations>0){
		for(ic=0; ic<HouseholdNlogL.Observations; ic++){
			if(HouseholdNlogL.StudyYear[ic]==CurrYear){
				if(SexInd==0){
					HouseholdNlogL.ModelPrev[ic] = 1.0 - HIVstageSumM[HouseholdNlogL.
						AgeStart[ic]][0]/TotalPopSum[HouseholdNlogL.AgeStart[ic]][0];}
				else{
					HouseholdNlogL.ModelPrev[ic] = 1.0 - HIVstageSumF[HouseholdNlogL.
						AgeStart[ic]][0]/TotalPopSum[HouseholdNlogL.AgeStart[ic]][1];}
			}
		}
	}

	if(HouseholdLogL.Observations>0){
		for(ic=0; ic<HouseholdLogL.Observations; ic++){
			if(HouseholdLogL.StudyYear[ic]==CurrYear){
				numerator=0;
				denominator=0;
				for(ia=HouseholdLogL.AgeStart[ic]; ia<HouseholdLogL.AgeEnd[ic]+1; ia++){
					if(SexInd==0){
						numerator += HIVstageSumM[ia][0];
						denominator += TotalPopSum[ia][0];
					}
					else{
						numerator += HIVstageSumF[ia][0];
						denominator += TotalPopSum[ia][1];
					}
				}
				HouseholdLogL.ModelPrev[ic] = 1.0 - numerator/denominator;
			}
		}
	}
}

NonHIVtransition::NonHIVtransition(){}

void NonHIVtransition::CalcProbCure()
{
	double PropnTreatedPublic, PropnTreatedPrivate;
	int iy;
	
	if(SexInd==0){
		PropnTreatedPublic = PropnTreatedPublicM;
		PropnTreatedPrivate = PropnTreatedPrivateM;}
	else{
		PropnTreatedPublic = PropnTreatedPublicF;
		PropnTreatedPrivate = PropnTreatedPrivateF;}
	iy = CurrYear - StartYear;
	
	ProbCompleteCure = (PropnTreatedPublic * (PropnPublicUsingSM[iy] * CorrectRxWithSM +
		(1.0 - PropnPublicUsingSM[iy]) * CorrectRxPreSM) * (1.0 - DrugShortage[iy]) +
		PropnTreatedPrivate * (PropnPrivateUsingSM[iy] * CorrectRxWithSM +
		(1.0 - PropnPrivateUsingSM[iy]) * CorrectRxPreSM)) * DrugEff + TradnalEff *
		(1.0 - PropnTreatedPublic - PropnTreatedPrivate);
}

void NonHIVtransition::ClearAlive()
{
	int ia, iz;

	for(ia=0; ia<16; ia++){
		for(iz=0; iz<nStates; iz++){
			AliveStage0[ia][iz] = 0;
			if(HIVind==1){
				AliveStage1[ia][iz] = 0;
				AliveStage2[ia][iz] = 0;
				AliveStage3[ia][iz] = 0;
				AliveStage4[ia][iz] = 0;
				AliveStage5[ia][iz] = 0;
			}
		}
	}
}

void NonHIVtransition::CalcTotalAlive()
{
	int ia, iz;

	for(ia=0; ia<16; ia++){
		for(iz=0; iz<nStates; iz++){
			AliveSum[ia][iz] = AliveStage0[ia][iz];
			if(HIVind==1){
				AliveSum[ia][iz] += AliveStage1[ia][iz] + AliveStage2[ia][iz] +
					AliveStage3[ia][iz] + AliveStage4[ia][iz] + AliveStage5[ia][iz];}
		}
	}
}

TradnalSTDtransition::TradnalSTDtransition(){}

void TradnalSTDtransition::GetANCprev()
{
	int ic, ia;
	double numerator;

	if(AntenatalNlogL.Observations>0){
		for(ic=0; ic<AntenatalNlogL.Observations; ic++){
			if(AntenatalNlogL.StudyYear[ic]==CurrYear){
				numerator = (AliveStage0[AntenatalNlogL.AgeStart[ic]][0] - 
					VirginsSum[AntenatalNlogL.AgeStart[ic]][1]) * 
					SexuallyExpFert[AntenatalNlogL.AgeStart[ic]-1];
				if(HIVind==1){
					numerator += (AliveStage1[AntenatalNlogL.AgeStart[ic]][0] * 
						RelHIVfertility[0] + AliveStage2[AntenatalNlogL.AgeStart[ic]][0] * 
						RelHIVfertility[1] + AliveStage3[AntenatalNlogL.AgeStart[ic]][0] * 
						RelHIVfertility[2] + AliveStage4[AntenatalNlogL.AgeStart[ic]][0] * 
						RelHIVfertility[3] + AliveStage5[AntenatalNlogL.AgeStart[ic]][0] * 
						RelHIVfertility[4]) * SexuallyExpFert[AntenatalNlogL.AgeStart[ic]-1];
				}
				AntenatalNlogL.ModelPrev[ic] = 1.0 - numerator/
					BirthsByAge[AntenatalNlogL.AgeStart[ic]-1];
			}
		}
	}

	if(ANClogL.Observations>0){
		for(ic=0; ic<ANClogL.Observations; ic++){
			if(ANClogL.StudyYear[ic]==CurrYear){
				numerator = 0;
				for(ia=0; ia<7; ia++){
					numerator += (AliveStage0[ia+1][0] - VirginsSum[ia+1][1]) * 
						SexuallyExpFert[ia];}
				if(HIVind==1){
					for(ia=0; ia<7; ia++){
						numerator += (AliveStage1[ia+1][0] * RelHIVfertility[0] + 
							AliveStage2[ia+1][0] * RelHIVfertility[1] + AliveStage3[ia+1][0] * 
							RelHIVfertility[2] + AliveStage4[ia+1][0] * RelHIVfertility[3] +
							AliveStage5[ia+1][0] * RelHIVfertility[4]) * SexuallyExpFert[ia];}
				}
				ANClogL.ModelPrev[ic] = 1.0 - numerator/TotalBirths;
			}
		}
	}
}

void TradnalSTDtransition::GetHHprev()
{
	// Similar to the GetHHprev function in the HIVtransition class, except that we are
	// not considering the case of national data (since there are as yet no national
	// household surveys of STIs other than HIV).

	int ic, ia;
	double numerator, denominator;

	if(HouseholdLogL.Observations>0){
		for(ic=0; ic<HouseholdLogL.Observations; ic++){
			if(HouseholdLogL.StudyYear[ic]==CurrYear){
				numerator=0;
				denominator=0;
				for(ia=HouseholdLogL.AgeStart[ic]; ia<HouseholdLogL.AgeEnd[ic]+1; ia++){
					numerator += AliveSum[ia][0];
					denominator += TotalPopSum[ia][SexInd];
				}
				HouseholdLogL.ModelPrev[ic] = 1.0 - numerator/denominator;
			}
		}
	}
}

void TradnalSTDtransition::GetFPCprev()
{
	int ia, ic;
	double numerator, denominator;

	if(FPClogL.Observations>0){
		for(ic=0; ic<FPClogL.Observations; ic++){
			if(FPClogL.StudyYear[ic]==CurrYear){
				numerator = 0;
				denominator = 0;
				for(ia=0; ia<16; ia++){
					numerator += AliveSum[ia][0] * FPCweights[ia];
					denominator += TotalPopSum[ia][1] * FPCweights[ia];
				}
				FPClogL.ModelPrev[ic] = 1.0 - numerator/denominator;
			}
		}
	}
}

SyphilisTransition::SyphilisTransition(int Sex, int ObsANC, int ObsFPC, int ObsGUD, 
									   int ObsCSW, int ObsHH, int ObsANCN, int ObsHHN)
{
	SexInd = Sex;
	nStates = 7;

	ANClogL.Observations = ObsANC;
	FPClogL.Observations = ObsFPC;
	GUDlogL.Observations = ObsGUD;
	CSWlogL.Observations = ObsCSW;
	HouseholdLogL.Observations = ObsHH;
	AntenatalNlogL.Observations = ObsANCN;
	HouseholdNlogL.Observations = ObsHHN;
}

void SyphilisTransition::CalcTransitionProbs()
{
	int ia;
	double RxRate, TeenRxRate, RednAsympDurFSW;
	double Adj2ndary; // Adjustment to prob of cure (for primary syphilis) in individuals 
					  // with 2ndary syphilis

	if(SexInd==0){
		RxRate = MaleRxRate;
		TeenRxRate = MaleTeenRxRate;}
	else{
		RxRate = FemRxRate;
		TeenRxRate = FemTeenRxRate;
		RednAsympDurFSW = 1.0/(1.0 + 1.0/(FSWasympRxRate * ProbCompleteCure * FSWasympCure *
			AveDuration[3]));}

	ANCpropnCured = ANCpropnScreened * ANCpropnTreated * DrugEff;
	ProbANCcured[0] = 0;
	if(SexInd==1){
		for(ia=1; ia<8; ia++){
			ProbANCcured[ia] = (1.0 - pow(1.0 - SexuallyExpFert[ia-1], 1.0/CycleD))*
				ANCpropnCured;}
	}

	From1to2 = 1.0 - exp(-(1.0/AveDuration[0]) * 52.0/CycleD);
	From2to3 = (1.0 - exp(-(1.0/AveDuration[1]) * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-RxRate * ProbCompleteCure * 52.0/CycleD)));
	From2to3T = (1.0 - exp(-(1.0/AveDuration[1]) * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-TeenRxRate * ProbCompleteCure * 52.0/CycleD)));
	From2to3C = (1.0 - exp(-(1.0/AveDuration[1]) * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-FSWRxRate * ProbCompleteCure * 52.0/CycleD)));
	From2to5 = (1.0 - exp(-RxRate * ProbCompleteCure * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-(1.0/AveDuration[1]) * 52.0/CycleD)));
	From2to5T = (1.0 - exp(-TeenRxRate * ProbCompleteCure * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-(1.0/AveDuration[1]) * 52.0/CycleD)));
	From2to5C = (1.0 - exp(-FSWRxRate * ProbCompleteCure * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-(1.0/AveDuration[1]) * 52.0/CycleD)));
	Adj2ndary = (1.0 - SecondaryRxMult) * (1.0 - SecondaryCureMult);
	From3to4 = (1.0 - exp(-(1.0/AveDuration[2]) * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-RxRate * ProbCompleteCure * Adj2ndary * 52.0/CycleD)));
	From3to4T = (1.0 - exp(-(1.0/AveDuration[2]) * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-TeenRxRate * ProbCompleteCure * Adj2ndary * 52.0/CycleD)));
	From3to4C = (1.0 - exp(-(1.0/AveDuration[2]) * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-FSWRxRate * ProbCompleteCure * Adj2ndary * 52.0/CycleD)));
	From3to5 = (1.0 - exp(-RxRate * ProbCompleteCure * Adj2ndary * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-(1.0/AveDuration[2]) * 52.0/CycleD)));
	From3to5T = (1.0 - exp(-TeenRxRate * ProbCompleteCure * Adj2ndary * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-(1.0/AveDuration[2]) * 52.0/CycleD)));
	From3to5C = (1.0 - exp(-FSWRxRate * ProbCompleteCure * Adj2ndary * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-(1.0/AveDuration[2]) * 52.0/CycleD)));
	From4to6 = 1.0 - exp(-(1.0/AveDuration[3]) * 52.0/CycleD);
	From4to6C = 1.0 - exp(-(1.0/(AveDuration[3] * (1 - RednAsympDurFSW))) * 52.0/CycleD);
	From5to0 = 1.0 - exp(-(1.0/AveDuration[4]) * 52.0/CycleD);
	From6to0 = 1.0 - exp(-(1.0/AveDuration[5]) * 52.0/CycleD);
}

void SyphilisTransition::GetGUDprev()
{
	int ia, ic;
	double numerator;

	if(GUDlogL.Observations>0){
		for(ic=0; ic<GUDlogL.Observations; ic++){
			if(CurrYear==GUDlogL.StudyYear[ic]){
				numerator = 0;
				for(ia=0; ia<16; ia++){
					numerator += AliveSum[ia][2];}
				GUDlogL.ModelPrev[ic] = numerator/TotalGUDcases[SexInd];
			}
		}
	}
}

HerpesTransition::HerpesTransition(int Sex, int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, 
								   int ObsHH, int ObsANCN, int ObsHHN)
{
	SexInd = Sex;
	nStates = 5;

	ANClogL.Observations = ObsANC;
	FPClogL.Observations = ObsFPC;
	GUDlogL.Observations = ObsGUD;
	CSWlogL.Observations = ObsCSW;
	HouseholdLogL.Observations = ObsHH;
	AntenatalNlogL.Observations = ObsANCN;
	HouseholdNlogL.Observations = ObsHHN;
}

void HerpesTransition::CalcTransitionProbs()
{
	int is;
	double RxRate, TeenRxRate;

	if(SexInd==0){
		RxRate = MaleRxRate;
		TeenRxRate = MaleTeenRxRate;}
	else{
		RxRate = FemRxRate;
		TeenRxRate = FemTeenRxRate;}

	From1to2 = 1.0 - exp(-((1.0/AveDuration[0]) + RxRate * ProbCompleteCure)*52.0/CycleD);
	From1to2T = 1.0 - exp(-((1.0/AveDuration[0]) + TeenRxRate * ProbCompleteCure)*
		52.0/CycleD);
	From2to3[0] = (1.0 - exp(-RecurrenceRate*52.0/CycleD)) *
		(1.0 - 0.5*(1.0 - exp(-(1.0/AveDuration[1])*52.0/CycleD)));
	for(is=1; is<6; is++){
		From2to3[is] = 1.0 - exp(-RecurrenceRate * (1.0 + HSVrecurrenceIncrease[is-1])*
			52.0/CycleD);}
	From3to2 = 1.0 - exp(-((1.0/AveDuration[2]) + RxRate * ProbCompleteCure)*52.0/CycleD);
	From3to2T = 1.0 - exp(-((1.0/AveDuration[2]) + TeenRxRate * ProbCompleteCure)*
		52.0/CycleD);
	From2to4 = (1 - exp(-(1.0/AveDuration[1])*52.0/CycleD)) *
		(1.0 - 0.5*(1.0 - exp(-RecurrenceRate*52.0/CycleD)));
	if(SexInd==1){
		From1to2C = 1.0 - exp(-((1.0/AveDuration[0]) + FSWRxRate * ProbCompleteCure)*
			52.0/CycleD);
		From3to2C = 1.0 - exp(-((1.0/AveDuration[2]) + FSWRxRate * ProbCompleteCure)*
			52.0/CycleD);}
}

void HerpesTransition::GetGUDprev()
{
	int ia, ic;
	double numerator;

	if(GUDlogL.Observations>0){
		for(ic=0; ic<GUDlogL.Observations; ic++){
			if(CurrYear==GUDlogL.StudyYear[ic]){
				numerator = 0;
				for(ia=0; ia<16; ia++){
					numerator += AliveSum[ia][1] + AliveSum[ia][3];}
				GUDlogL.ModelPrev[ic] = numerator/TotalGUDcases[SexInd];
			}
		}
	}
}

OtherSTDtransition::OtherSTDtransition(int Sex, int ObsANC, int ObsFPC, int ObsGUD, 
									   int ObsCSW, int ObsHH, int ObsANCN, int ObsHHN)
{
	SexInd = Sex;
	nStates = 3;

	ANClogL.Observations = ObsANC;
	FPClogL.Observations = ObsFPC;
	GUDlogL.Observations = ObsGUD;
	CSWlogL.Observations = ObsCSW;
	HouseholdLogL.Observations = ObsHH;
	AntenatalNlogL.Observations = ObsANCN;
	HouseholdNlogL.Observations = ObsHHN;
}

void OtherSTDtransition::CalcTransitionProbs()
{
	double RxRate, TeenRxRate, RednAsympDurFSW;

	if(SexInd==0){
		RxRate = MaleRxRate;
		TeenRxRate = MaleTeenRxRate;}
	else{
		RxRate = FemRxRate;
		TeenRxRate = FemTeenRxRate;
		RednAsympDurFSW = 1.0/(1.0 + 1.0/(FSWasympRxRate * ProbCompleteCure * FSWasympCure *
			AveDuration[1]));}

	From1to0 = 1.0 - exp(-((1.0/AveDuration[0]) + RxRate * ProbCompleteCure)*52.0/CycleD);
	From1to0T = 1.0 - exp(-((1.0/AveDuration[0]) + TeenRxRate * ProbCompleteCure)*
		52.0/CycleD);
	From2to0 = 1.0 - exp(-(1.0/AveDuration[1])*52.0/CycleD);
	// Transition probabilities for sex workers
	if(SexInd==1){
		From1to0C = 1.0 - exp(-((1.0/AveDuration[0]) + FSWRxRate * ProbCompleteCure)*
			52.0/CycleD);
		From2to0C = 1.0 - exp(-(1.0/(AveDuration[1]*(1.0 - RednAsympDurFSW)))*52.0/CycleD);}
}

void OtherSTDtransition::GetGUDprev()
{
	int ia, ic;
	double numerator;

	if(GUDlogL.Observations>0){
		for(ic=0; ic<GUDlogL.Observations; ic++){
			if(CurrYear==GUDlogL.StudyYear[ic]){
				numerator = 0;
				for(ia=0; ia<16; ia++){
					numerator += AliveSum[ia][1];}
				GUDlogL.ModelPrev[ic] = numerator/TotalGUDcases[SexInd];
			}
		}
	}
}

NonSTDtransition::NonSTDtransition(){}

void NonSTDtransition::CalcProbPartialCure()
{
	int iy;
	
	iy = CurrYear - StartYear;
	
	ProbPartialCure = (PropnTreatedPublicF * (PropnPublicUsingSM[iy] * CorrectRxWithSM +
		(1.0 - PropnPublicUsingSM[iy]) * CorrectRxPreSM) * (1.0 - DrugShortage[iy]) +
		PropnTreatedPrivateF * (PropnPrivateUsingSM[iy] * CorrectRxWithSM +
		(1.0 - PropnPrivateUsingSM[iy]) * CorrectRxPreSM)) * DrugPartialEff + TradnalEff *
		(1.0 - PropnTreatedPublicF - PropnTreatedPrivateF);
}

BVtransition::BVtransition(int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, int ObsHH, 
							int ObsANCN, int ObsHHN)
{
	SexInd = 1;
	nStates = 4;

	ANClogL.Observations = ObsANC;
	FPClogL.Observations = ObsFPC;
	GUDlogL.Observations = ObsGUD;
	CSWlogL.Observations = ObsCSW;
	HouseholdLogL.Observations = ObsHH;
	AntenatalNlogL.Observations = ObsANCN;
	HouseholdNlogL.Observations = ObsHHN;
}

void BVtransition::CalcTransitionProbs()
{
	int ij;
	double RednAsympDurFSW;

	RednAsympDurFSW = 1.0/(1.0 + 1.0/(FSWasympRxRate * (ProbCompleteCure + ProbPartialCure) * 
		FSWasympCure * AveDuration[3]));
	From1to2 = 1.0 - exp(-CtsTransition[0][1] * 52.0/CycleD);
	From2to1ind = 1.0 - exp(-CtsTransition[1][0] * 52.0/CycleD);
	From3to1 = (1.0 - exp(-(CtsTransition[2][0] + FemRxRate * ProbCompleteCure) * 52.0/
		CycleD)) * (1.0 - 0.5*(1.0 - exp(-(CtsTransition[2][1] + FemRxRate * 
		ProbPartialCure) * 52.0/CycleD)));
	From3to1T = (1.0 - exp(-(CtsTransition[2][0] + FemTeenRxRate * ProbCompleteCure) * 52.0/
		CycleD)) * (1.0 - 0.5*(1.0 - exp(-(CtsTransition[2][1] + FemTeenRxRate * 
		ProbPartialCure) * 52.0/CycleD)));
	From3to1C = (1.0 - exp(-(CtsTransition[2][0] + FSWRxRate * ProbCompleteCure) * 52.0/
		CycleD)) * (1.0 - 0.5*(1.0 - exp(-(CtsTransition[2][1] + FSWRxRate * 
		ProbPartialCure) * 52.0/CycleD)));
	From3to2 = (1.0 - exp(-(CtsTransition[2][1] + FemRxRate * ProbPartialCure) * 52.0/
		CycleD)) * (1.0 - 0.5*(1.0 - exp(-(CtsTransition[2][0] + FemRxRate * 
		ProbCompleteCure) * 52.0/CycleD)));
	From3to2 = (1.0 - exp(-(CtsTransition[2][1] + FemRxRate * ProbPartialCure) * 52.0/
		CycleD)) * (1.0 - 0.5*(1.0 - exp(-(CtsTransition[2][0] + FemRxRate * 
		ProbCompleteCure) * 52.0/CycleD)));
	From3to2T = (1.0 - exp(-(CtsTransition[2][1] + FemTeenRxRate * ProbPartialCure) * 52.0/
		CycleD)) * (1.0 - 0.5*(1.0 - exp(-(CtsTransition[2][0] + FemTeenRxRate * 
		ProbCompleteCure) * 52.0/CycleD)));
	From3to2C = (1.0 - exp(-(CtsTransition[2][1] + FSWRxRate * ProbPartialCure) * 52.0/
		CycleD)) * (1.0 - 0.5*(1.0 - exp(-(CtsTransition[2][0] + FSWRxRate * 
		ProbCompleteCure) * 52.0/CycleD)));
	From4to1 = (1.0 - exp(-CtsTransition[3][0] * 52.0/CycleD)) *
		(1.0 - 0.5 * (1.0 - exp(-CtsTransition[3][1] * 52.0/CycleD)));
	From4to1C = (1.0 - exp(-(CtsTransition[3][0]/(1.0 - RednAsympDurFSW))*52.0/CycleD)) *
		(1.0 - 0.5 * (1.0 - exp(-(CtsTransition[3][1]/(1.0 - RednAsympDurFSW))*52.0/CycleD)));
	From4to2 = (1.0 - exp(-CtsTransition[3][1] * 52.0/CycleD)) *
		(1.0 - 0.5 * (1.0 - exp(-CtsTransition[3][0] * 52.0/CycleD)));
	From4to2C = (1.0 - exp(-(CtsTransition[3][1]/(1.0 - RednAsympDurFSW))*52.0/CycleD)) *
		(1.0 - 0.5 * (1.0 - exp(-(CtsTransition[3][0]/(1.0 - RednAsympDurFSW))*52.0/CycleD)));

	From2to3ind[0] = 1.0 - exp(-CtsTransition[1][2] * (1.0 - IncidenceMultNoPartners)*52.0/
		CycleD);
	From2to3ind[1] = 1.0 - exp(-CtsTransition[1][2] * 52.0/CycleD);
	From2to3ind[2] = 1.0 - exp(-CtsTransition[1][2] * (1.0 + IncidenceMultTwoPartners)*52.0/
		CycleD);
	From2to4ind[0] = 1.0 - exp(-CtsTransition[1][3] * (1.0 - IncidenceMultNoPartners)*52.0/
		CycleD);
	From2to4ind[1] = 1.0 - exp(-CtsTransition[1][3] * 52.0/CycleD);
	From2to4ind[2] = 1.0 - exp(-CtsTransition[1][3] * (1.0 + IncidenceMultTwoPartners)*52.0/
		CycleD);
	for(ij=0; ij<3; ij++){
		From2to1dep[ij] = From2to1ind * (1.0 - 0.5 * (From2to3ind[ij] + From2to4ind[ij]) +
			From2to3ind[ij] * From2to4ind[ij]/3.0);
		From2to3dep[ij] = From2to3ind[ij] * (1.0 - 0.5 * (From2to1ind + From2to4ind[ij]) +
			From2to1ind * From2to4ind[ij]/3.0);
		From2to4dep[ij] = From2to4ind[ij] * (1.0 - 0.5 * (From2to1ind + From2to3ind[ij]) +
			From2to1ind * From2to3ind[ij]/3.0);
	}
}

void BVtransition::GetANCprev()
{
	// Same as the function in the TradnalSTDtransition class, except that (a) I haven't
	// included the part for calculating national ANC prevalence (since there aren't any
	// national surveys that measure BV prevalence), (b) the formula for calculating
	// numbers of births to HIV-negative women with no BV is different, and (c) the 
	// calculation has to take into account that BV stages 1 and 2 are both 'negative'.

	int ic, ia;
	double numerator;

	if(ANClogL.Observations>0){
		for(ic=0; ic<ANClogL.Observations; ic++){
			if(ANClogL.StudyYear[ic]==CurrYear){
				numerator = 0;
				for(ia=0; ia<7; ia++){
					numerator += (AliveStage0[ia+1][0] + AliveStage0[ia+1][1] - 
						FemHigh.Virgin.FBV.NumberByStage0[ia+1][0] - 
						FemLow.Virgin.FBV.NumberByStage0[ia+1][0] - 
						FemHigh.Virgin.FBV.NumberByStage0[ia+1][1] -
						FemLow.Virgin.FBV.NumberByStage0[ia+1][1]) * SexuallyExpFert[ia];}
				if(HIVind==1){
					for(ia=0; ia<7; ia++){
						numerator += ((AliveStage1[ia+1][0] + AliveStage1[ia+1][1]) * 
							RelHIVfertility[0] + (AliveStage2[ia+1][0] + AliveStage2[ia+1][1]) * 
							RelHIVfertility[1] + (AliveStage3[ia+1][0] + AliveStage3[ia+1][1]) * 
							RelHIVfertility[2] + (AliveStage4[ia+1][0] + AliveStage4[ia+1][1]) * 
							RelHIVfertility[3] + (AliveStage5[ia+1][0] + AliveStage5[ia+1][1]) * 
							RelHIVfertility[4]) * SexuallyExpFert[ia];}
				}
				ANClogL.ModelPrev[ic] = 1.0 - numerator/TotalBirths;
			}
		}
	}
}

void BVtransition::GetFPCprev()
{
	// Similar to the GetFPCprev function in the TradnalSTDtransition class, except that
	// we take into account that both BV states 1 and 2 are considered 'negative'.

	int ia, ic;
	double numerator, denominator;

	if(FPClogL.Observations>0){
		for(ic=0; ic<FPClogL.Observations; ic++){
			if(FPClogL.StudyYear[ic]==CurrYear){
				numerator = 0;
				denominator = 0;
				for(ia=0; ia<16; ia++){
					numerator += (AliveSum[ia][0] + AliveSum[ia][1]) * FPCweights[ia];
					denominator += TotalPopSum[ia][1] * FPCweights[ia];
				}
				FPClogL.ModelPrev[ic] = 1.0 - numerator/denominator;
			}
		}
	}
}

VCtransition::VCtransition(int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, int ObsHH, 
							 int ObsANCN, int ObsHHN)
{
	SexInd = 1;
	nStates = 3;

	ANClogL.Observations = ObsANC;
	FPClogL.Observations = ObsFPC;
	GUDlogL.Observations = ObsGUD;
	CSWlogL.Observations = ObsCSW;
	HouseholdLogL.Observations = ObsHH;
	AntenatalNlogL.Observations = ObsANCN;
	HouseholdNlogL.Observations = ObsHHN;
}

void VCtransition::CalcTransitionProbs()
{
	int ia, is;
	double RednAsympDurFSW; // Although it's not strictly necessary to include this (since
							// we set it to 0), we may want to change this in future, so I
							// have included it.

	RednAsympDurFSW = 0.0;
	From1to2 = (1.0 - exp(-RecurrenceRate * 52.0/CycleD)) * 
		(1.0 - 0.5 * (1.0 - exp(-(1.0/AveDuration[0]) * 52.0/CycleD)));
	From1to2C = (1.0 - exp(-RecurrenceRate * 52.0/CycleD)) * (1.0 - 0.5 *
		(1.0 - exp(-(1.0/(AveDuration[0] * (1.0 - RednAsympDurFSW))) * 52.0/CycleD)));
	From1to0 = (1.0 - exp(-(1.0/AveDuration[0]) * 52.0/CycleD)) *
		(1.0 - 0.5 * (1.0 - exp(-RecurrenceRate * 52.0/CycleD)));
	From1to0C = (1.0 - exp(-(1.0/(AveDuration[0] * (1.0 - RednAsympDurFSW))) * 52.0/CycleD))*
		(1.0 - 0.5 * (1.0 - exp(-RecurrenceRate * 52.0/CycleD)));
	From2to1 = (1.0 - exp(-((1.0/AveDuration[1]) + FemRxRate * ProbPartialCure)*52.0/
		CycleD))*(1.0 - 0.5 * (1.0 - exp(-FemRxRate * ProbCompleteCure * 52.0/CycleD)));
	From2to1T = (1.0 - exp(-((1.0/AveDuration[1]) + FemTeenRxRate * ProbPartialCure)*52.0/
		CycleD))*(1.0 - 0.5 * (1.0 - exp(-FemTeenRxRate * ProbCompleteCure * 52.0/CycleD)));
	From2to1C = (1.0 - exp(-((1.0/AveDuration[1]) + FSWRxRate * ProbPartialCure)*52.0/
		CycleD))*(1.0 - 0.5 * (1.0 - exp(-FSWRxRate * ProbCompleteCure * 52.0/CycleD)));
	From2to0 = (1.0 - exp(-FemRxRate * ProbCompleteCure * 52.0/CycleD)) * (1.0 - 0.5 * 
		(1.0 - exp(-((1.0/AveDuration[1]) + FemRxRate * ProbPartialCure)*52.0/CycleD)));
	From2to0T = (1.0 - exp(-FemTeenRxRate * ProbCompleteCure * 52.0/CycleD)) * (1.0 - 0.5 * 
		(1.0 - exp(-((1.0/AveDuration[1]) + FemTeenRxRate * ProbPartialCure)*52.0/CycleD)));
	From2to0C = (1.0 - exp(-FSWRxRate * ProbCompleteCure * 52.0/CycleD)) * (1.0 - 0.5 * 
		(1.0 - exp(-((1.0/AveDuration[1]) + FSWRxRate * ProbPartialCure)*52.0/CycleD)));

	for(ia=0; ia<7; ia++){
		From0to1[ia][0] = 1.0 - exp(-Incidence * (HIVnegFert[ia]/HIVnegFert[0])*52.0/CycleD);}
	if(HIVind==1){
		for(ia=0; ia<7; ia++){
			for(is=1; is<6; is++){
				From0to1[ia][is] = 1.0 - exp(-Incidence * (HIVnegFert[ia]/HIVnegFert[0]) *
					(1.0 + IncidenceIncrease[is-1]) * 52.0/CycleD);}
		}
	}
}

void VCtransition::GetANCprev()
{
	// Same as the function in the TradnalSTDtransition class, except that (a) I haven't
	// included the part for calculating national ANC prevalence (since there aren't any
	// national surveys that measure candidiasis prevalence), and (b) the formula for
	// calculating numbers of births to HIV-negative women with no candidiasis is different.

	int ic, ia;
	double numerator;

	if(ANClogL.Observations>0){
		for(ic=0; ic<ANClogL.Observations; ic++){
			if(ANClogL.StudyYear[ic]==CurrYear){
				numerator = 0;
				for(ia=0; ia<7; ia++){
					numerator += (AliveStage0[ia+1][0] - FemHigh.Virgin.FVC.
						NumberByStage0[ia+1][0] - FemLow.Virgin.FVC.NumberByStage0[ia+1][0]) * 
						SexuallyExpFert[ia];}
				if(HIVind==1){
					for(ia=0; ia<7; ia++){
						numerator += (AliveStage1[ia+1][0] * RelHIVfertility[0] + 
							AliveStage2[ia+1][0] * RelHIVfertility[1] + AliveStage3[ia+1][0] * 
							RelHIVfertility[2] + AliveStage4[ia+1][0] * RelHIVfertility[3] +
							AliveStage5[ia+1][0] * RelHIVfertility[4]) * SexuallyExpFert[ia];}
				}
				ANClogL.ModelPrev[ic] = 1.0 - numerator/TotalBirths;
			}
		}
	}
}

void VCtransition::GetFPCprev()
{
	// Identical to the GetFPCprev function in the TradnalSTDtransition class

	int ia, ic;
	double numerator, denominator;

	if(FPClogL.Observations>0){
		for(ic=0; ic<FPClogL.Observations; ic++){
			if(FPClogL.StudyYear[ic]==CurrYear){
				numerator = 0;
				denominator = 0;
				for(ia=0; ia<16; ia++){
					numerator += AliveSum[ia][0] * FPCweights[ia];
					denominator += TotalPopSum[ia][1] * FPCweights[ia];
				}
				FPClogL.ModelPrev[ic] = 1.0 - numerator/denominator;
			}
		}
	}
}

STD::STD(){}

NonHIV::NonHIV(){}

void NonHIV::GetTotalBySTDstage(NonHIVtransition* a)
{
	int ia, iz;

	for(ia=0; ia<16; ia++){
		for(iz=0; iz<nStates; iz++){
			a->AliveStage0[ia][iz] += NumberByStage0[ia][iz];
			if(HIVind==1){
				a->AliveStage1[ia][iz] += NumberByStage1[ia][iz];
				a->AliveStage2[ia][iz] += NumberByStage2[ia][iz];
				a->AliveStage3[ia][iz] += NumberByStage3[ia][iz];
				a->AliveStage4[ia][iz] += NumberByStage4[ia][iz];
				a->AliveStage5[ia][iz] += NumberByStage5[ia][iz];
			}
		}
	}
}

NonSTD::NonSTD(){}

BV::BV()
{
	nStates = 4;
}

void BV::CalcSTDtransitions(BVtransition* a, int FSWind, int Partners)
{
	CalcSTDstageTransitions(PropnByStage0, TempPropnByStage0, a, FSWind, Partners);
	if(HIVind==1){
		CalcSTDstageTransitions(PropnByStage1, TempPropnByStage1, a, FSWind, Partners);
		CalcSTDstageTransitions(PropnByStage2, TempPropnByStage2, a, FSWind, Partners);
		CalcSTDstageTransitions(PropnByStage3, TempPropnByStage3, a, FSWind, Partners);
		CalcSTDstageTransitions(PropnByStage4, TempPropnByStage4, a, FSWind, Partners);
		CalcSTDstageTransitions(PropnByStage5, TempPropnByStage5, a, FSWind, Partners);
	}
}

void BV::CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7], 
								 BVtransition* a, int FSWind, int Partners)
{
	int ia;

	if(FSWind==1){Partners=2;}

	for(ia=0; ia<16; ia++){ // Do the calcs that differ by # partners
		TempPropn[ia][0] = Propn[ia][0] * (1.0 - a->From1to2) + Propn[ia][1] * 
			a->From2to1dep[Partners];
		TempPropn[ia][1] = Propn[ia][0] * a->From1to2 + Propn[ia][1] * (1.0 - 
			a->From2to1dep[Partners] - a->From2to3dep[Partners] - a->From2to4dep[Partners]);
		TempPropn[ia][2] = Propn[ia][1] * a->From2to3dep[Partners];
		TempPropn[ia][3] = Propn[ia][1] * a->From2to4dep[Partners];
	}
	if(FSWind==0){ // Do the calcs that differ by age
		for(ia=0; ia<2; ia++){
			TempPropn[ia][0] += Propn[ia][2] * a->From3to1T + Propn[ia][3] * a->From4to1;
			TempPropn[ia][1] += Propn[ia][2] * a->From3to2T + Propn[ia][3] * a->From4to2;
			TempPropn[ia][2] += Propn[ia][2] * (1.0 - a->From3to1T - a->From3to2T);
			TempPropn[ia][3] += Propn[ia][3] * (1.0 - a->From4to1 - a->From4to2);
		}
		for(ia=2; ia<16; ia++){
			TempPropn[ia][0] += Propn[ia][2] * a->From3to1 + Propn[ia][3] * a->From4to1;
			TempPropn[ia][1] += Propn[ia][2] * a->From3to2 + Propn[ia][3] * a->From4to2;
			TempPropn[ia][2] += Propn[ia][2] * (1.0 - a->From3to1 - a->From3to2);
			TempPropn[ia][3] += Propn[ia][3] * (1.0 - a->From4to1 - a->From4to2);
		}
	}
	else{ // Do the calcs that differ in sex workers
		for(ia=0; ia<16; ia++){
			TempPropn[ia][0] += Propn[ia][2] * a->From3to1C + Propn[ia][3] * a->From4to1C;
			TempPropn[ia][1] += Propn[ia][2] * a->From3to2C + Propn[ia][3] * a->From4to2C;
			TempPropn[ia][2] += Propn[ia][2] * (1.0 - a->From3to1C - a->From3to2C);
			TempPropn[ia][3] += Propn[ia][3] * (1.0 - a->From4to1C - a->From4to2C);
		}
	}
}

VC::VC()
{
	nStates = 3;
}

void VC::CalcSTDtransitions(VCtransition* a, int FSWind)
{
	CalcSTDstageTransitions(PropnByStage0, TempPropnByStage0, a, FSWind, 0);
	if(HIVind==1){
		CalcSTDstageTransitions(PropnByStage1, TempPropnByStage1, a, FSWind, 1);
		CalcSTDstageTransitions(PropnByStage2, TempPropnByStage2, a, FSWind, 2);
		CalcSTDstageTransitions(PropnByStage3, TempPropnByStage3, a, FSWind, 3);
		CalcSTDstageTransitions(PropnByStage4, TempPropnByStage4, a, FSWind, 4);
		CalcSTDstageTransitions(PropnByStage5, TempPropnByStage5, a, FSWind, 5);
	}
}

void VC::CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7],
								 VCtransition* a, int FSWind, int is)
{
	int ia;

	for(ia=0; ia<16; ia++){ // Do the calcs that differ by HIV stage
		if(ia>0 && ia<=7){
			TempPropn[ia][0] = 0 - Propn[ia][0] * a->From0to1[ia-1][is];
			TempPropn[ia][1] = Propn[ia][0] * a->From0to1[ia-1][is];
		}
		else{
			TempPropn[ia][0] = 0;
			TempPropn[ia][1] = 0;
		}
	}
	if(FSWind==0){ // Do the calcs that differ by age
		for(ia=0; ia<2; ia++){
			TempPropn[ia][0] += Propn[ia][0] + Propn[ia][1] * a->From1to0 + Propn[ia][2] *
				a->From2to0T;
			TempPropn[ia][1] += Propn[ia][1] * (1.0 - a->From1to0 - a->From1to2) + 
				Propn[ia][2] * a->From2to1T;
			TempPropn[ia][2] = Propn[ia][1] * a->From1to2 + Propn[ia][2] * (1.0 - 
				a->From2to0T - a->From2to1T);
		}
		for(ia=2; ia<16; ia++){
			TempPropn[ia][0] += Propn[ia][0] + Propn[ia][1] * a->From1to0 + Propn[ia][2] *
				a->From2to0;
			TempPropn[ia][1] += Propn[ia][1] * (1.0 - a->From1to0 - a->From1to2) + 
				Propn[ia][2] * a->From2to1;
			TempPropn[ia][2] = Propn[ia][1] * a->From1to2 + Propn[ia][2] * (1.0 - 
				a->From2to0 - a->From2to1);
		}
	}
	else{ // Do the calcs that differ for sex workers
		for(ia=0; ia<16; ia++){
			TempPropn[ia][0] += Propn[ia][0] + Propn[ia][1] * a->From1to0C + Propn[ia][2] *
				a->From2to0C;
			TempPropn[ia][1] += Propn[ia][1] * (1.0 - a->From1to0C - a->From1to2C) + 
				Propn[ia][2] * a->From2to1C;
			TempPropn[ia][2] = Propn[ia][1] * a->From1to2C + Propn[ia][2] * (1.0 - 
				a->From2to0C - a->From2to1C);
		}
	}
}

TradnalSTD::TradnalSTD(){}

Syphilis::Syphilis()
{
	nStates = 7;
}

void Syphilis::CalcTransmissionProb(SyphilisTransition* a)
{
	int ia;
	double ProbIfInfected;

	ProbIfInfected = a->TransmProb;
	for(ia=0; ia<16; ia++){
		if(HIVind==1){
			WeightedProbTransm[ia] = ProbIfInfected * (NumberByStage0[ia][1] + 
				NumberByStage0[ia][2] + NumberByStage0[ia][3] + NumberByStage1[ia][1] + 
				NumberByStage1[ia][2] + NumberByStage1[ia][3] + NumberByStage2[ia][1] + 
				NumberByStage2[ia][2] + NumberByStage2[ia][3] + NumberByStage3[ia][1] + 
				NumberByStage3[ia][2] + NumberByStage3[ia][3] + NumberByStage4[ia][1] + 
				NumberByStage4[ia][2] + NumberByStage4[ia][3] + NumberByStage5[ia][1] + 
				NumberByStage5[ia][2] + NumberByStage5[ia][3]);}
		else{
			WeightedProbTransm[ia] = ProbIfInfected * (NumberByStage0[ia][1] +  
				NumberByStage0[ia][2] + NumberByStage0[ia][3]);}
	}
}

void Syphilis::CalcSTDtransitions(SyphilisTransition* a, int FSWind)
{
	CalcSTDstageTransitions(PropnByStage0, TempPropnByStage0, a, FSWind);
	if(HIVind==1){
		CalcSTDstageTransitions(PropnByStage1, TempPropnByStage1, a, FSWind);
		CalcSTDstageTransitions(PropnByStage2, TempPropnByStage2, a, FSWind);
		CalcSTDstageTransitions(PropnByStage3, TempPropnByStage3, a, FSWind);
		CalcSTDstageTransitions(PropnByStage4, TempPropnByStage4, a, FSWind);
		CalcSTDstageTransitions(PropnByStage5, TempPropnByStage5, a, FSWind);
	}
}

void Syphilis::CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7],
									   SyphilisTransition* a, int FSWind)
{
	int ia;

	for(ia=0; ia<16; ia++){ // Do calcs that are the same for all ages & risk groups
		TempPropn[ia][0] = Propn[ia][0] * (1.0 - InfectProb[ia]);
		if(SyphilisImmunity==0){
			TempPropn[ia][0] += (Propn[ia][5]*a->From5to0 + Propn[ia][6]*a->From6to0) 
				* (1.0 - InfectProb[ia]);}
		else{
			TempPropn[ia][0] += Propn[ia][5]*a->From5to0 + Propn[ia][6]*a->From6to0;}
		TempPropn[ia][1] = Propn[ia][0] * InfectProb[ia] + Propn[ia][1] * (1.0 - 
			a->From1to2) * (1.0 - a->ProbANCcured[ia]);
		if(SyphilisImmunity==0){
			TempPropn[ia][1] += (Propn[ia][5]+ Propn[ia][6]) * InfectProb[ia];}
		if(SyphilisImmunity==1){
			TempPropn[ia][5] = Propn[ia][5] * (1.0 - a->From5to0);
			TempPropn[ia][6] = Propn[ia][6] * (1.0 - a->From6to0);
		}
		else{
			TempPropn[ia][5] = Propn[ia][5] * (1.0 - a->From5to0) *
				(1.0 - InfectProb[ia]);
			TempPropn[ia][6] = Propn[ia][6] * (1.0 - a->From6to0) *
				(1.0 - InfectProb[ia]);
		}
		if(FSWind==0){
			TempPropn[ia][6] += Propn[ia][4] * (a->ProbANCcured[ia] + a->From4to6 *
				(1.0 - a->ProbANCcured[ia]));}
	}

	if(FSWind==0){ // Do calculations that differ by age
		for(ia=0; ia<2; ia++){
			TempPropn[ia][2] = (Propn[ia][1] * a->From1to2 + Propn[ia][2] * 
				(1.0 - a->From2to3T - a->From2to5T)) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][3] = (Propn[ia][2] * a->From2to3T + Propn[ia][3] *
				(1.0 - a->From3to4T - a->From3to5T)) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][4] = (Propn[ia][3] * a->From3to4T + Propn[ia][4] *
				(1.0 - a->From4to6)) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][5] += (Propn[ia][1] + Propn[ia][2] + Propn[ia][3]) * 
				a->ProbANCcured[ia] + (Propn[ia][2] * a->From2to5T + Propn[ia][3] *
				a->From3to5T) * (1.0 - a->ProbANCcured[ia]);
		}
		for(ia=2; ia<16; ia++){
			TempPropn[ia][2] = (Propn[ia][1] * a->From1to2 + Propn[ia][2] * 
				(1.0 - a->From2to3 - a->From2to5)) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][3] = (Propn[ia][2] * a->From2to3 + Propn[ia][3] *
				(1.0 - a->From3to4 - a->From3to5)) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][4] = (Propn[ia][3] * a->From3to4 + Propn[ia][4] *
				(1.0 - a->From4to6)) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][5] += (Propn[ia][1] + Propn[ia][2] + Propn[ia][3]) * 
				a->ProbANCcured[ia] + (Propn[ia][2] * a->From2to5 + Propn[ia][3] *
				a->From3to5) * (1.0 - a->ProbANCcured[ia]);
		}
	}
	else{ // Do calculations that differ among sex workers
		for(ia=0; ia<16; ia++){
			TempPropn[ia][2] = (Propn[ia][1] * a->From1to2 + Propn[ia][2] * 
				(1.0 - a->From2to3C - a->From2to5C)) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][3] = (Propn[ia][2] * a->From2to3C + Propn[ia][3] *
				(1.0 - a->From3to4C - a->From3to5C)) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][4] = (Propn[ia][3] * a->From3to4C + Propn[ia][4] *
				(1.0 - a->From4to6C)) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][5] += (Propn[ia][1] + Propn[ia][2] + Propn[ia][3]) * 
				a->ProbANCcured[ia] + (Propn[ia][2] * a->From2to5C + Propn[ia][3] *
				a->From3to5C) * (1.0 - a->ProbANCcured[ia]);
			TempPropn[ia][6] += Propn[ia][4] * (a->ProbANCcured[ia] + a->From4to6C *
				(1.0 - a->ProbANCcured[ia]));
		}
	}
}

MaleTP::MaleTP(){}

FemTP::FemTP(){}

Herpes::Herpes()
{
	nStates = 5;
}

void Herpes::CalcTransmissionProb(HerpesTransition* a)
{
	int ia;
	double ProbIfInfected;

	ProbIfInfected = a->TransmProb;
	for(ia=0; ia<16; ia++){
		WeightedProbTransm[ia] = ProbIfInfected * (HSVsymptomInfecIncrease *
			(NumberByStage0[ia][1] + NumberByStage0[ia][3]) + NumberByStage0[ia][2] + 
			NumberByStage0[ia][4]);
		if(HIVind==1){
			WeightedProbTransm[ia] += ProbIfInfected * ((HSVsymptomInfecIncrease *
				(NumberByStage1[ia][1] + NumberByStage1[ia][3]) + NumberByStage1[ia][2] + 
				NumberByStage1[ia][4]) * (1.0 + HSVsheddingIncrease[0]) + (HSVsymptomInfecIncrease *
				(NumberByStage2[ia][1] + NumberByStage2[ia][3]) + NumberByStage2[ia][2] + 
				NumberByStage2[ia][4]) * (1.0 + HSVsheddingIncrease[1]) + (HSVsymptomInfecIncrease *
				(NumberByStage3[ia][1] + NumberByStage3[ia][3]) + NumberByStage3[ia][2] + 
				NumberByStage3[ia][4]) * (1.0 + HSVsheddingIncrease[2]) + (HSVsymptomInfecIncrease *
				(NumberByStage4[ia][1] + NumberByStage4[ia][3]) + NumberByStage4[ia][2] + 
				NumberByStage4[ia][4]) * (1.0 + HSVsheddingIncrease[3]) + (HSVsymptomInfecIncrease *
				(NumberByStage5[ia][1] + NumberByStage5[ia][3]) + NumberByStage5[ia][2] + 
				NumberByStage5[ia][4]) * (1.0 + HSVsheddingIncrease[4]));
		}
	}
}

void Herpes::CalcSTDtransitions(HerpesTransition* a, int FSWind)
{
	CalcSTDstageTransitions(PropnByStage0, TempPropnByStage0, a, FSWind, 0);
	if(HIVind==1){
		CalcSTDstageTransitions(PropnByStage1, TempPropnByStage1, a, FSWind, 1);
		CalcSTDstageTransitions(PropnByStage2, TempPropnByStage2, a, FSWind, 2);
		CalcSTDstageTransitions(PropnByStage3, TempPropnByStage3, a, FSWind, 3);
		CalcSTDstageTransitions(PropnByStage4, TempPropnByStage4, a, FSWind, 4);
		CalcSTDstageTransitions(PropnByStage5, TempPropnByStage5, a, FSWind, 5);
	}
}

void Herpes::CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7],
									 HerpesTransition* a, int FSWind, int is)
{
	int ia;

	for(ia=0; ia<16; ia++){ // Do calcs that are the same for all ages and risk groups
		TempPropn[ia][0] = Propn[ia][0] * (1.0 - InfectProb[ia]);
		if(is==0){
			TempPropn[ia][2] = Propn[ia][2] * (1.0 - a->From2to3[0] - a->From2to4);
			TempPropn[ia][3] = Propn[ia][2] * a->From2to3[0];
			TempPropn[ia][4] = Propn[ia][4] + Propn[ia][2] * a->From2to4;
		}
		else{
			TempPropn[ia][2] = Propn[ia][2] * (1.0 - a->From2to3[is]);
			TempPropn[ia][3] = Propn[ia][2] * a->From2to3[is];
			TempPropn[ia][4] = Propn[ia][4];
		}
		TempPropn[ia][4] += Propn[ia][0] * InfectProb[ia] * (1.0 - a->SymptomaticPropn);
	}
	if(FSWind==0){ // Do calculations that differ by age
		for(ia=0; ia<2; ia++){
			TempPropn[ia][1] = Propn[ia][0] * InfectProb[ia] * a->SymptomaticPropn +
				Propn[ia][1] * (1.0 - a->From1to2T);
			TempPropn[ia][2] += Propn[ia][1] * a->From1to2T + Propn[ia][3] * a->From3to2T;
			TempPropn[ia][3] += Propn[ia][3] * (1.0 - a->From3to2T);
		}
		for(ia=2; ia<16; ia++){
			TempPropn[ia][1] = Propn[ia][0] * InfectProb[ia] * a->SymptomaticPropn +
				Propn[ia][1] * (1.0 - a->From1to2);
			TempPropn[ia][2] += Propn[ia][1] * a->From1to2 + Propn[ia][3] * a->From3to2;
			TempPropn[ia][3] += Propn[ia][3] * (1.0 - a->From3to2);
		}
	}
	else{ // Do calculations that differ for sex workers
		for(ia=0; ia<16; ia++){
			TempPropn[ia][1] = Propn[ia][0] * InfectProb[ia] * a->SymptomaticPropn +
				Propn[ia][1] * (1.0 - a->From1to2C);
			TempPropn[ia][2] += Propn[ia][1] * a->From1to2C + Propn[ia][3] * a->From3to2C;
			TempPropn[ia][3] += Propn[ia][3] * (1.0 - a->From3to2C);
		}
	}
}

MaleHSV::MaleHSV(){}

FemHSV::FemHSV(){}

OtherSTD::OtherSTD()
{
	nStates = 3;
}

void OtherSTD::CalcTransmissionProb(OtherSTDtransition* a)
{
	int ia;
	double ProbIfInfected;

	ProbIfInfected = a->TransmProb;
	for(ia=0; ia<16; ia++){
		WeightedProbTransm[ia] = ProbIfInfected * (NumberByStage0[ia][1] + 
			NumberByStage0[ia][2]);
		if(HIVind==1){
			WeightedProbTransm[ia] += ProbIfInfected * (NumberByStage1[ia][1] + 
				NumberByStage1[ia][2] + NumberByStage2[ia][1] + NumberByStage2[ia][2] + 
				NumberByStage3[ia][1] + NumberByStage3[ia][2] + NumberByStage4[ia][1] + 
				NumberByStage4[ia][2] + NumberByStage5[ia][1] + NumberByStage5[ia][2]);
		}
	}
}

void OtherSTD::CalcSTDtransitions(OtherSTDtransition* a, int FSWind)
{
	CalcSTDstageTransitions(PropnByStage0, TempPropnByStage0, a, FSWind);
	if(HIVind==1){
		CalcSTDstageTransitions(PropnByStage1, TempPropnByStage1, a, FSWind);
		CalcSTDstageTransitions(PropnByStage2, TempPropnByStage2, a, FSWind);
		CalcSTDstageTransitions(PropnByStage3, TempPropnByStage3, a, FSWind);
		CalcSTDstageTransitions(PropnByStage4, TempPropnByStage4, a, FSWind);
		CalcSTDstageTransitions(PropnByStage5, TempPropnByStage5, a, FSWind);
	}
}

void OtherSTD::CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7],
									   OtherSTDtransition* a, int FSWind)
{
	int ia;

	if(FSWind==0){
		for(ia=0; ia<2; ia++){
			TempPropn[ia][0] = Propn[ia][0] * (1.0 - InfectProb[ia]) + Propn[ia][1] * 
				a->From1to0T + Propn[ia][2] * a->From2to0;
			TempPropn[ia][1] = Propn[ia][0] * InfectProb[ia] * a->SymptomaticPropn + 
				Propn[ia][1] * (1.0 - a->From1to0T);
			TempPropn[ia][2] = Propn[ia][0] * InfectProb[ia] * (1.0 - a->SymptomaticPropn) + 
				Propn[ia][2] * (1.0 - a->From2to0);
		}
		for(ia=2; ia<16; ia++){
			TempPropn[ia][0] = Propn[ia][0] * (1.0 - InfectProb[ia]) + Propn[ia][1] * 
				a->From1to0 + Propn[ia][2] * a->From2to0;
			TempPropn[ia][1] = Propn[ia][0] * InfectProb[ia] * a->SymptomaticPropn + 
				Propn[ia][1] * (1.0 - a->From1to0);
			TempPropn[ia][2] = Propn[ia][0] * InfectProb[ia] * (1.0 - a->SymptomaticPropn) + 
				Propn[ia][2] * (1.0 - a->From2to0);
		}
	}
	else{ // FSW case
		for(ia=0; ia<16; ia++){
			TempPropn[ia][0] = Propn[ia][0] * (1.0 - InfectProb[ia]) + Propn[ia][1] * 
				a->From1to0C + Propn[ia][2] * a->From2to0C;
			TempPropn[ia][1] = Propn[ia][0] * InfectProb[ia] * a->SymptomaticPropn + 
				Propn[ia][1] * (1.0 - a->From1to0C);
			TempPropn[ia][2] = Propn[ia][0] * InfectProb[ia] * (1.0 - a->SymptomaticPropn) + 
				Propn[ia][2] * (1.0 - a->From2to0C);
		}
	}
}

MaleHD::MaleHD(){}

FemHD::FemHD(){}

MaleNG::MaleNG(){}

FemNG::FemNG(){}

MaleCT::MaleCT(){}

FemCT::FemCT(){}

MaleTV::MaleTV(){}

FemTV::FemTV(){}

PaedHIV::PaedHIV(){}

void PaedHIV::CalcAIDSprogression()
{
	int ia;

	AIDSprob1st6m = 1.0 - pow(0.5, pow(0.5/PreAIDSmedian, PreAIDSshape));
	for(ia=0; ia<15; ia++){
		AIDSprogressionRate[ia] = 1.0 - pow(0.5, pow((ia + 1.5)/PreAIDSmedian, PreAIDSshape)
			- pow((ia + 0.5)/PreAIDSmedian, PreAIDSshape));
	}
}

void PaedHIV::CalcAgeChanges()
{
	int ia, yr;

	yr = CurrYear - StartYear;

	// First calculate AIDS deaths (by age that would have been attained at end of yr)
	AIDSdeaths[0] = NewHIV * AIDSprob1st6m * (1.0 - HAARTaccess[yr]) *
		(1.0 - exp(-0.25/MeanAIDSsurvival)) * (1.0 - MortProb1st6m);
	for(ia=1; ia<15; ia++){
		AIDSdeaths[ia] = (AIDSstage[ia-1] * (1.0 - exp(-1.0/MeanAIDSsurvival)) +
			PreAIDSstage[ia-1] * AIDSprogressionRate[ia-1] * (1.0 - HAARTaccess[yr]) *
			(1.0 - exp(-0.5/MeanAIDSsurvival))) * (1.0 - NonAIDSmort[ia-1]);
	}

	// Then calculate age changes
	for(ia=0; ia<14; ia++){
		PreAIDSstage[14-ia] = PreAIDSstage[13-ia] * (1.0 - AIDSprogressionRate[13-ia]) *
			(1.0 - NonAIDSmort[13-ia]);
		AIDSstage[14-ia] = (AIDSstage[13-ia] * exp(-1.0/MeanAIDSsurvival) +
			PreAIDSstage[13-ia] * AIDSprogressionRate[13-ia] * (1.0 - HAARTaccess[yr]) *
			exp(-0.5/MeanAIDSsurvival)) * (1.0 - NonAIDSmort[13-ia]);
	}
	PreAIDSstage[0] = NewHIV * (1.0 - AIDSprob1st6m) * (1.0 - MortProb1st6m);
	AIDSstage[0] = NewHIV * AIDSprob1st6m * (1.0 - HAARTaccess[yr]) *
		exp(-0.25/MeanAIDSsurvival) * (1.0 - MortProb1st6m);
}

SexBehavGrp::SexBehavGrp(){}

SexuallyExp::SexuallyExp(){}

void SexuallyExp::SetPartnerNumbers(int S1, int S2, int L1, int L2)
{
	NumberPartnersS1 = S1;
	NumberPartnersS2 = S2;
	NumberPartnersL1 = L1;
	NumberPartnersL2 = L2;
	NumberPartners = S1 + S2 + L1 + L2;
	if(L1 > 0 || L2 > 0){
		MarriedInd = 1;}
	if(SexInd==0 && RiskGroup==1){
		if(NumberPartners==0){FSWcontactOffset=0;}
		if(NumberPartners==1){
			if(S1==1 || S2==1){FSWcontactOffset=1;}
			else{FSWcontactOffset=2;}
		}
		if(NumberPartners==2){
			if(S1+S2==2){FSWcontactOffset=3;}
			else{FSWcontactOffset=4;}
		}
	}
	VirginInd = 0;
}

void SexuallyExp::GetNumbersBySTDstage(NonHIV* m)
{
	int ia, iz;

	for(ia=0; ia<16; ia++){
		for(iz=0; iz<m->nStates; iz++){
			m->NumberByStage0[ia][iz] = NumbersByHIVstage[ia][0] * m->PropnByStage0[ia][iz];
			if(HIVind==1){
				m->NumberByStage1[ia][iz] = NumbersByHIVstage[ia][1]*m->PropnByStage1[ia][iz];
				m->NumberByStage2[ia][iz] = NumbersByHIVstage[ia][2]*m->PropnByStage2[ia][iz];
				m->NumberByStage3[ia][iz] = NumbersByHIVstage[ia][3]*m->PropnByStage3[ia][iz];
				m->NumberByStage4[ia][iz] = NumbersByHIVstage[ia][4]*m->PropnByStage4[ia][iz];
				m->NumberByStage5[ia][iz] = NumbersByHIVstage[ia][5]*m->PropnByStage5[ia][iz];
			}
		}
	}
}

void SexuallyExp::GetPropnsBySTDstage(NonHIV* m)
{
	int ia, iz;

	for(ia=0; ia<16; ia++){
		for(iz=0; iz<m->nStates; iz++){
			if(NumbersByHIVstage[ia][0]!=0){
				m->PropnByStage0[ia][iz] = m->NumberByStage0[ia][iz]/
					NumbersByHIVstage[ia][0];}
			if(HIVind==1){
				if(NumbersByHIVstage[ia][1]!=0){
					m->PropnByStage1[ia][iz] = m->NumberByStage1[ia][iz]/
						NumbersByHIVstage[ia][1];}
				if(NumbersByHIVstage[ia][2]!=0){
					m->PropnByStage2[ia][iz] = m->NumberByStage2[ia][iz]/
						NumbersByHIVstage[ia][2];}
				if(NumbersByHIVstage[ia][3]!=0){
					m->PropnByStage3[ia][iz] = m->NumberByStage3[ia][iz]/
						NumbersByHIVstage[ia][3];}
				if(NumbersByHIVstage[ia][4]!=0){
					m->PropnByStage4[ia][iz] = m->NumberByStage4[ia][iz]/
						NumbersByHIVstage[ia][4];}
				if(NumbersByHIVstage[ia][5]!=0){
					m->PropnByStage5[ia][iz] = m->NumberByStage5[ia][iz]/
						NumbersByHIVstage[ia][5];}
			}
		}
	}
}

void SexuallyExp::GetSTDcofactor(NonHIV* m, NonHIVtransition* a)
{
	int ia, iz;
	double CofactorSum;

	for(ia=0; ia<16; ia++){
		CofactorSum = 0;
		for(iz=1; iz<m->nStates; iz++){
			CofactorSum += m->PropnByStage0[ia][iz] * a->HIVsuscepIncrease[iz-1];}
		STDcofactor[ia][0] *= (1.0 + CofactorSum);
		CofactorSum = 0;
		for(iz=1; iz<m->nStates; iz++){
			CofactorSum += m->PropnByStage1[ia][iz] * a->HIVinfecIncrease[iz-1];}
		STDcofactor[ia][1] *= (1.0 + CofactorSum);
		CofactorSum = 0;
		for(iz=1; iz<m->nStates; iz++){
			CofactorSum += m->PropnByStage2[ia][iz] * a->HIVinfecIncrease[iz-1];}
		STDcofactor[ia][2] *= (1.0 + CofactorSum);
		CofactorSum = 0;
		for(iz=1; iz<m->nStates; iz++){
			CofactorSum += m->PropnByStage3[ia][iz] * a->HIVinfecIncrease[iz-1];}
		STDcofactor[ia][3] *= (1.0 + CofactorSum);
		CofactorSum = 0;
		for(iz=1; iz<m->nStates; iz++){
			CofactorSum += m->PropnByStage4[ia][iz] * a->HIVinfecIncrease[iz-1];}
		STDcofactor[ia][4] *= (1.0 + CofactorSum);
		CofactorSum = 0;
		for(iz=1; iz<m->nStates; iz++){
			CofactorSum += m->PropnByStage5[ia][iz] * a->HIVinfecIncrease[iz-1];}
		STDcofactor[ia][5] *= (1.0 + CofactorSum);
	}
}

void SexuallyExp::CalcHIVtransmProb(HIVtransition* a)
{
	int ia;

	if(RiskGroup==1){
		for(ia=0; ia<16; ia++){
			a->TransmProbS1to1[ia] += WeightedProbTransm[ia][1] * NumberPartnersS1;
			a->TransmProbS1to2[ia] += WeightedProbTransm[ia][2] * NumberPartnersS2;
			a->TransmProbL1to1[ia] += WeightedProbTransm[ia][4] * NumberPartnersL1;
			a->TransmProbL1to2[ia] += WeightedProbTransm[ia][5] * NumberPartnersL2;
		}
		if(SexInd==0){
			WeightedProbTransmFSW = 0;
			for(ia=0; ia<16; ia++){
				WeightedProbTransmFSW += WeightedProbTransm[ia][0] * AgeEffectFSWcontact[ia];}
			WeightedProbTransmFSW *= PartnerEffectFSWcontact[FSWcontactOffset] *
				FSWcontactConstant;
			a->InfectProbFSW[0] += WeightedProbTransmFSW;
		}
		else{
			if(FSWind==1){
				WeightedProbTransmFSW = 0;
				for(ia=0; ia<16; ia++){
					WeightedProbTransmFSW += WeightedProbTransm[ia][0];}
				a->InfectProbFSW[0] = WeightedProbTransmFSW;
			}
		}
	}
	else{
		for(ia=0; ia<16; ia++){
			a->TransmProbS2to1[ia] += WeightedProbTransm[ia][2] * NumberPartnersS1;
			a->TransmProbS2to2[ia] += WeightedProbTransm[ia][3] * NumberPartnersS2;
			a->TransmProbL2to1[ia] += WeightedProbTransm[ia][5] * NumberPartnersL1;
			a->TransmProbL2to2[ia] += WeightedProbTransm[ia][6] * NumberPartnersL2;
		}
	}
}

void SexuallyExp::CalcHIVtransitions(HIVtransition* a)
{
	int ia;

	for(ia=0; ia<16; ia++){
		HIVstageExits[ia][0] = NumbersByHIVstage[ia][0] * HIVinfectProb[ia];
		HIVstageExits[ia][1] = NumbersByHIVstage[ia][1] * a->From1to2;
		HIVstageExits[ia][2] = NumbersByHIVstage[ia][2] * a->From2to3;
		HIVstageExits[ia][3] = NumbersByHIVstage[ia][3] * (a->From3to4 + a->From3to5);
		HIVstageExits[ia][4] = NumbersByHIVstage[ia][4] * a->From4toDead;
		HIVstageExits[ia][5] = NumbersByHIVstage[ia][5] * a->From5toDead;
		if(FixedUncertainty==1){
			NewHIVsum[ia][SexInd] += HIVstageExits[ia][0];
			/*if(MarriedInd==1){
				NewHIVsumML[ia][SexInd] += NumbersByHIVstage[ia][0] * HIVinfectProbLT[ia] *
					(1.0 - 0.5 * (HIVinfectProbST[ia] + HIVinfectProbCS[ia]) +
					HIVinfectProbST[ia] * HIVinfectProbCS[ia]/3.0);
				NewHIVsumMS[ia][SexInd] += NumbersByHIVstage[ia][0] * HIVinfectProbST[ia] *
					(1.0 - 0.5 * (HIVinfectProbLT[ia] + HIVinfectProbCS[ia]) +
					HIVinfectProbLT[ia] * HIVinfectProbCS[ia]/3.0);
			}
			if(MarriedInd==0){
				NewHIVsumUS[ia][SexInd] += NumbersByHIVstage[ia][0] * HIVinfectProbST[ia] *
					(1.0 - 0.5 * HIVinfectProbCS[ia]);}
			if(RiskGroup==1){
				NewHIVsumCS[ia][SexInd] += NumbersByHIVstage[ia][0] * HIVinfectProbCS[ia] *
					(1.0 - 0.5 * (HIVinfectProbST[ia] + HIVinfectProbLT[ia]) +
					HIVinfectProbST[ia] * HIVinfectProbLT[ia]/3.0);}*/
			/*if(MarriedInd==1 && CurrYear==2007){
				NewHIVsumM[ia][SexInd] += HIVstageExits[ia][0];}
			if(MarriedInd==0 && CurrYear==2007){
				NewHIVsumU[ia][SexInd] += HIVstageExits[ia][0];}*/
			NewAIDSdeaths[ia][SexInd] += HIVstageExits[ia][4] + HIVstageExits[ia][5];
		}
		TempNumbersByHIVstage[ia][0] = NumbersByHIVstage[ia][0] - HIVstageExits[ia][0];
		TempNumbersByHIVstage[ia][1] = NumbersByHIVstage[ia][1] - HIVstageExits[ia][1] +
			HIVstageExits[ia][0];
		TempNumbersByHIVstage[ia][2] = NumbersByHIVstage[ia][2] - HIVstageExits[ia][2] +
			HIVstageExits[ia][1];
		TempNumbersByHIVstage[ia][3] = NumbersByHIVstage[ia][3] - HIVstageExits[ia][3] +
			HIVstageExits[ia][2];
		TempNumbersByHIVstage[ia][4] = NumbersByHIVstage[ia][4] - HIVstageExits[ia][4] +
			NumbersByHIVstage[ia][3] * a->From3to4;
		TempNumbersByHIVstage[ia][5] = NumbersByHIVstage[ia][5] - HIVstageExits[ia][5] +
			NumbersByHIVstage[ia][3] * a->From3to5;
	}
}

void SexuallyExp::HIVstageChanges(NonHIV* a)
{
	int ia, id;
	double HAARTpropn;

	HAARTpropn = HAARTaccess[CurrYear-StartYear];
	for(ia=0; ia<16; ia++){
		for(id=0; id<a->nStates; id++){
			a->PropnByStage0[ia][id] = a->TempPropnByStage0[ia][id];}
		if(HIVind==1){
			if(TempNumbersByHIVstage[ia][1]!=0){
				for(id=0; id<a->nStates; id++){
					a->PropnByStage1[ia][id] = (HIVstageExits[ia][0] * a->TempPropnByStage0[ia][id]
						+ (NumbersByHIVstage[ia][1] - HIVstageExits[ia][1]) * 
						a->TempPropnByStage1[ia][id])/TempNumbersByHIVstage[ia][1];}
			}
			if(TempNumbersByHIVstage[ia][2]!=0){
				for(id=0; id<a->nStates; id++){
					a->PropnByStage2[ia][id] = (HIVstageExits[ia][1] * a->TempPropnByStage1[ia][id]
						+ (NumbersByHIVstage[ia][2] - HIVstageExits[ia][2]) * 
						a->TempPropnByStage2[ia][id])/TempNumbersByHIVstage[ia][2];}
			}
			if(TempNumbersByHIVstage[ia][3]!=0){
				for(id=0; id<a->nStates; id++){
					a->PropnByStage3[ia][id] = (HIVstageExits[ia][2] * a->TempPropnByStage2[ia][id]
						+ (NumbersByHIVstage[ia][3] - HIVstageExits[ia][3]) * 
						a->TempPropnByStage3[ia][id])/TempNumbersByHIVstage[ia][3];}
			}
			if(TempNumbersByHIVstage[ia][4]!=0){
				for(id=0; id<a->nStates; id++){
					a->PropnByStage4[ia][id] = (HIVstageExits[ia][3] * a->TempPropnByStage3[ia][id] *
						(1.0 - HAARTpropn) + (NumbersByHIVstage[ia][4] - HIVstageExits[ia][4]) 
						* a->TempPropnByStage4[ia][id])/TempNumbersByHIVstage[ia][4];}
			}
			if(TempNumbersByHIVstage[ia][5]!=0){
				for(id=0; id<a->nStates; id++){
					a->PropnByStage5[ia][id] = (HIVstageExits[ia][3] * a->TempPropnByStage3[ia][id]
						* HAARTpropn + (NumbersByHIVstage[ia][5] - HIVstageExits[ia][5]) * 
						a->TempPropnByStage5[ia][id])/TempNumbersByHIVstage[ia][5];}
			}
		}
	}
}

void SexuallyExp::SetHIVnumbersToTemp()
{
	// Note that it isn't necessary to call this function at the end of the STD cycle if  
	// HIVind=0, since the TempNumbersByHIVstage array is only calculated if HIVind=1.
	// However, it is necessary to call the function at the end of every sexual behav cycle.

	int ia, is;

	for(ia=0; ia<16; ia++){
		NumbersByHIVstage[ia][0] = TempNumbersByHIVstage[ia][0];
		TotalAlive[ia] = TempNumbersByHIVstage[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				NumbersByHIVstage[ia][is] = TempNumbersByHIVstage[ia][is];
				TotalAlive[ia] += TempNumbersByHIVstage[ia][is];
			}
		}
	}
}

void SexuallyExp::GetNewPartners()
{
	int ia, is;
	double BasePartners;

	// First calculate DesiredNewPartners
	if(VirginInd==0){
		if(NumberPartners==0 || (NumberPartners==1 && RiskGroup==1)){
			BasePartners = PartnershipFormation[RiskGroup-1][SexInd];
			if(NumberPartnersS1 + NumberPartnersS2 == 1){
				BasePartners *= PartnerEffectNew[0][SexInd];}
			else if(NumberPartnersL1 + NumberPartnersL2 == 1){
				BasePartners *= PartnerEffectNew[1][SexInd];}
			DesiredNewPartners = 0;
			for(ia=0; ia<16; ia++){
				DesiredNewPartners += NumbersByHIVstage[ia][0] * BasePartners * 
					AgeEffectPartners[ia][SexInd];
				if(HIVind==1){
					for(is=1; is<6; is++){
						DesiredNewPartners += NumbersByHIVstage[ia][is] * BasePartners * 
							AgeEffectPartners[ia][SexInd] * HIVeffectPartners[is-1];}
				}
			}
		}
	}
	else{
		DesiredNewPartners = 0;
		if(RiskGroup==1){
			for(ia=0; ia<16; ia++){
				DesiredNewPartners += NumbersByHIVstage[ia][0] * SexualDebut[ia][SexInd];}
		}
		else{
			for(ia=0; ia<16; ia++){
				DesiredNewPartners += NumbersByHIVstage[ia][0] * SexualDebut[ia][SexInd] *
					DebutAdjLow[SexInd];}
		}
	}

	// Secondly, calculate DesiredNewL1 (desired new marriages to partners in risk group 1)
	if(NumberPartnersS1>0 && NumberPartnersL1==0 && NumberPartnersL2==0){
		BasePartners = MarriageRate[0][SexInd*2 + RiskGroup - 1];
		DesiredNewL1 = 0;
		for(ia=0; ia<16; ia++){
			DesiredNewL1 += TotalAlive[ia] * BasePartners * NumberPartnersS1
				* AgeEffectMarriage[ia][SexInd*2 + RiskGroup - 1];}
	}

	// Thirdly, calculate DesiredNewL2 (desired new marriages to partners in risk group 2)
	if(NumberPartnersS2>0 && NumberPartnersL1==0 && NumberPartnersL2==0){
		BasePartners = MarriageRate[1][SexInd*2 + RiskGroup - 1];
		DesiredNewL2 = 0;
		for(ia=0; ia<16; ia++){
			DesiredNewL2 += TotalAlive[ia] * BasePartners * NumberPartnersS2 
				* AgeEffectMarriage[ia][SexInd*2 + RiskGroup - 1];}
	}

	// Fourthly, calculate AnnFSWcontacts
	if(RiskGroup==1 && SexInd==0){
		AnnFSWcontacts = 0;
		for(ia=0; ia<16; ia++){
			AnnFSWcontacts += TotalAlive[ia] * AgeEffectFSWcontact[ia];}
		AnnFSWcontacts *= FSWcontactConstant * PartnerEffectFSWcontact[FSWcontactOffset];
	}
}

void SexuallyExp::GetPartnerTransitions()
{
	// Calculates independent probabilities of change in relationship status over current
	// sexual behaviour cycle

	int ia, is;
	double BasePartners, BasePartnersS1, BasePartnersS2;
	double BreakupRate, Denominator;

	// First calculate AcquireNewS1 and AcquireNewS2
	if(VirginInd==0){
		if(NumberPartners==0 || (NumberPartners==1 && RiskGroup==1)){
			BasePartners = PartnershipFormation[RiskGroup-1][SexInd];
			if(NumberPartnersS1 + NumberPartnersS2 == 1){
				BasePartners *= PartnerEffectNew[0][SexInd];}
			else if(NumberPartnersL1 + NumberPartnersL2 == 1){
				BasePartners *= PartnerEffectNew[1][SexInd];}
			if(SexInd==0){
				BasePartnersS1 = BasePartners * DesiredPartnerRiskM[RiskGroup-1][0] * 
					AdjSTrateM[RiskGroup-1][0]/CycleS;
				BasePartnersS2 = BasePartners * DesiredPartnerRiskM[RiskGroup-1][1] * 
					AdjSTrateM[RiskGroup-1][1]/CycleS;
			}
			else{
				BasePartnersS1 = BasePartners * DesiredPartnerRiskF[RiskGroup-1][0] * 
					AdjSTrateF[RiskGroup-1][0]/CycleS;
				BasePartnersS2 = BasePartners * DesiredPartnerRiskF[RiskGroup-1][1] * 
					AdjSTrateF[RiskGroup-1][1]/CycleS;
			}
			for(ia=0; ia<16; ia++){
				AcquireNewS1[ia][0] = 1.0 - exp(-BasePartnersS1 * AgeEffectPartners[ia][SexInd]);
				AcquireNewS2[ia][0] = 1.0 - exp(-BasePartnersS2 * AgeEffectPartners[ia][SexInd]);
			}
			if(HIVind==1){
				for(is=1; is<6; is++){
					for(ia=0; ia<16; ia++){
						AcquireNewS1[ia][is] = AcquireNewS1[ia][0] * HIVeffectPartners[is-1];
						AcquireNewS2[ia][is] = AcquireNewS2[ia][0] * HIVeffectPartners[is-1];
					}
				}
			}
		}
	}
	else{
		if(SexInd==0){
			BasePartnersS1 = DesiredPartnerRiskM[RiskGroup-1][0]/CycleS;
			BasePartnersS2 = DesiredPartnerRiskM[RiskGroup-1][1]/CycleS;
		}
		else{
			BasePartnersS1 = DesiredPartnerRiskF[RiskGroup-1][0]/CycleS;
			BasePartnersS2 = DesiredPartnerRiskF[RiskGroup-1][1]/CycleS;
		}
		for(ia=0; ia<16; ia++){
			if(RiskGroup==1){
				AcquireNewS1[ia][0] = 1.0 - exp(-BasePartnersS1 * SexualDebut[ia][SexInd]);
				AcquireNewS2[ia][0] = 1.0 - exp(-BasePartnersS2 * SexualDebut[ia][SexInd]);
			}
			else{
				AcquireNewS1[ia][0] = 1.0 - exp(-BasePartnersS1 * SexualDebut[ia][SexInd] *
					DebutAdjLow[SexInd]);
				AcquireNewS2[ia][0] = 1.0 - exp(-BasePartnersS2 * SexualDebut[ia][SexInd] *
					DebutAdjLow[SexInd]);
			}
		}
	}

	// Secondly, calculate MarryL1
	if(NumberPartnersS1>0 && NumberPartnersL1==0 && NumberPartnersL2==0){
		BasePartners = MarriageRate[0][SexInd*2 + RiskGroup - 1] * NumberPartnersS1/CycleS;
		if(SexInd==0){
			BasePartners *= AdjLTrateM[RiskGroup-1][0];}
		else{
			BasePartners *= AdjLTrateF[RiskGroup-1][0];}
		for(ia=0; ia<16; ia++){
			MarryL1[ia][0] = 1.0 - exp(-BasePartners * 
				AgeEffectMarriage[ia][SexInd*2 + RiskGroup - 1]);
			if(HIVind==1){
				for(is=1; is<6; is++){
					MarryL1[ia][is] = MarryL1[ia][0];}
			}
		}
	}

	// Thirdly, calculate MarryL2
	if(NumberPartnersS2>0 && NumberPartnersL1==0 && NumberPartnersL2==0){
		BasePartners = MarriageRate[1][SexInd*2 + RiskGroup - 1] * NumberPartnersS2/CycleS;
		if(SexInd==0){
			BasePartners *= AdjLTrateM[RiskGroup-1][1];}
		else{
			BasePartners *= AdjLTrateF[RiskGroup-1][1];}
		for(ia=0; ia<16; ia++){
			MarryL2[ia][0] = 1.0 - exp(-BasePartners * 
				AgeEffectMarriage[ia][SexInd*2 + RiskGroup - 1]);
			if(HIVind==1){
				for(is=1; is<6; is++){
					MarryL2[ia][is] = MarryL2[ia][0];}
			}
		}
	}
	
	// Fourthly, calculate LoseL
	// (a) for the case in which the LT partner is in risk group 1
	if(NumberPartnersL1>0){
		if(SexInd==0){
			for(ia=0; ia<16; ia++){
				LoseL[ia][0] = 1.0 - exp(-(LTseparation[ia][0] + NonAIDSmortPartner[ia][0] +
					AIDSmortPartnerM[ia][3+RiskGroup])/CycleS);}
		}
		else{
			for(ia=0; ia<16; ia++){
				LoseL[ia][0] = 1.0 - exp(-(LTseparation[ia][1] + NonAIDSmortPartner[ia][1] +
					AIDSmortPartnerF[ia][3+RiskGroup])/CycleS);}
		}
		if(HIVind==1){
			for(is=1; is<6; is++){
				for(ia=0; ia<16; ia++){
					LoseL[ia][is] = LoseL[ia][0];}
			}
		}
	}
	// (b) ...for the case in which the LT partner is in risk group 2
	if(NumberPartnersL2>0){
		if(SexInd==0){
			for(ia=0; ia<16; ia++){
				LoseL[ia][0] = 1.0 - exp(-(LTseparation[ia][0] + NonAIDSmortPartner[ia][0] +
					AIDSmortPartnerM[ia][5+RiskGroup])/CycleS);}
		}
		else{
			for(ia=0; ia<16; ia++){
				LoseL[ia][0] = 1.0 - exp(-(LTseparation[ia][1] + NonAIDSmortPartner[ia][1] +
					AIDSmortPartnerF[ia][5+RiskGroup])/CycleS);}
		}
		if(HIVind==1){
			for(is=1; is<6; is++){
				for(ia=0; ia<16; ia++){
					LoseL[ia][is] = LoseL[ia][0];}
			}
		}
	}

	// Fifthly, calculate LoseS1 (similar in format to code for calculating LoseL, part a)
	if(NumberPartnersS1>0){
		if(SexInd==0){
			BreakupRate = 1.0/MeanDurSTrel[RiskGroup-1][0];
			for(ia=0; ia<16; ia++){
				LoseS1[ia][0] = 1.0 - exp(-(BreakupRate + NonAIDSmortPartner[ia][0] +
					AIDSmortPartnerM[ia][RiskGroup-1]) * NumberPartnersS1/CycleS);}
		}
		else{
			BreakupRate = 1.0/MeanDurSTrel[0][RiskGroup-1];
			for(ia=0; ia<16; ia++){
				LoseS1[ia][0] = 1.0 - exp(-(BreakupRate + NonAIDSmortPartner[ia][1] +
					AIDSmortPartnerF[ia][RiskGroup-1]) * NumberPartnersS1/CycleS);}
		}
		if(HIVind==1){
			for(is=1; is<6; is++){
				for(ia=0; ia<16; ia++){
					LoseS1[ia][is] = LoseS1[ia][0];}
			}
		}
	}

	// Sixthly, calculate LoseS2 (similar in format to code for calculating LoseL, part b)
	if(NumberPartnersS2>0){
		if(SexInd==0){
			BreakupRate = 1.0/MeanDurSTrel[RiskGroup-1][1];
			for(ia=0; ia<16; ia++){
				LoseS2[ia][0] = 1.0 - exp(-(BreakupRate + NonAIDSmortPartner[ia][0] +
					AIDSmortPartnerM[ia][RiskGroup+1]) * NumberPartnersS2/CycleS);}
		}
		else{
			BreakupRate = 1.0/MeanDurSTrel[1][RiskGroup-1];
			for(ia=0; ia<16; ia++){
				LoseS2[ia][0] = 1.0 - exp(-(BreakupRate + NonAIDSmortPartner[ia][1] +
					AIDSmortPartnerF[ia][RiskGroup+1]) * NumberPartnersS2/CycleS);}
		}
		if(HIVind==1){
			for(is=1; is<6; is++){
				for(ia=0; ia<16; ia++){
					LoseS2[ia][is] = LoseS2[ia][0];}
			}
		}
	}

	// Seventhly, calculate EnterSW
	if(NumberPartners==0 && SexInd==1 && VirginInd==0){
		Denominator = 0;
		for(ia=0; ia<16; ia++){
			Denominator += NumbersByHIVstage[ia][0] * FSWentry[ia];}
		if(HIVind==1){
			for(is=1; is<6; is++){
				for(ia=0; ia<16; ia++){
					Denominator += NumbersByHIVstage[ia][is] * FSWentry[ia] * 
						HIVeffectFSWentry[is-1];}
			}
		}
		for(ia=0; ia<16; ia++){
			EnterSW[ia][0] = RequiredNewFSW * FSWentry[ia]/Denominator;
			if(EnterSW[ia][0]>1.0){
				EnterSW[ia][0] = 1.0;
				ErrorInd = 1;
			}
			if(HIVind==1){
				for(is=1; is<6; is++){
					EnterSW[ia][is] = RequiredNewFSW * FSWentry[ia] * HIVeffectFSWentry[is-1]/
						Denominator;
					if(EnterSW[ia][is]>1.0){
						EnterSW[ia][is] = 1.0;
						ErrorInd = 1;
					}
				}
			}
		}
		if(ErrorInd==1){ErrorCount += 1;}
	}

	// Lastly, calculate LeaveSW
	if(FSWind==1){
		for(ia=0; ia<16; ia++){
			LeaveSW[ia][0] = 1.0 - exp(-FSWexit[ia]/CycleS);
			if(HIVind==1){
				for(is=1; is<6; is++){
					LeaveSW[ia][is] = LeaveSW[ia][0] * HIVeffectFSWexit[is-1];}
			}
		}
	}
}

void SexuallyExp::ConvertToDependent1(double IndArray1[16][6], double DepArray1[16][6])
{
	int ia, is;

	for(ia=0; ia<16; ia++){
		DepArray1[ia][0] = IndArray1[ia][0] * (1.0 - NonAIDSmortProb[ia][SexInd]);
		Remain[ia][0] = 1.0 - NonAIDSmortProb[ia][SexInd] - DepArray1[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				DepArray1[ia][is] = IndArray1[ia][is] * (1.0 - NonAIDSmortProb[ia][SexInd]);
				Remain[ia][is] = 1.0 - NonAIDSmortProb[ia][SexInd] - DepArray1[ia][is];
			}
		}
	}
}

void SexuallyExp::ConvertToDependent2(double IndArray1[16][6], double IndArray2[16][6], 
		double DepArray1[16][6], double DepArray2[16][6])
{
	int ia, is;

	for(ia=0; ia<16; ia++){
		DepArray1[ia][0] = IndArray1[ia][0] * (1.0 - 0.5 * IndArray2[ia][0]) * 
			(1.0 - NonAIDSmortProb[ia][SexInd]);
		DepArray2[ia][0] = IndArray2[ia][0] * (1.0 - 0.5 * IndArray1[ia][0]) * 
			(1.0 - NonAIDSmortProb[ia][SexInd]);
		Remain[ia][0] = 1.0 - NonAIDSmortProb[ia][SexInd] - DepArray1[ia][0] - 
			DepArray2[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				DepArray1[ia][is] = IndArray1[ia][is] * (1.0 - 0.5 * IndArray2[ia][is]) * 
					(1.0 - NonAIDSmortProb[ia][SexInd]);
				DepArray2[ia][is] = IndArray2[ia][is] * (1.0 - 0.5 * IndArray1[ia][is]) * 
					(1.0 - NonAIDSmortProb[ia][SexInd]);
				Remain[ia][is] = 1.0 - NonAIDSmortProb[ia][SexInd] - DepArray1[ia][is] - 
					DepArray2[ia][is];
			}
		}
	}
}

void SexuallyExp::ConvertToDependent3(double IndArray1[16][6], double IndArray2[16][6],
		double IndArray3[16][6], double DepArray1[16][6], double DepArray2[16][6], 
		double DepArray3[16][6])
{
	int ia, is;

	for(ia=0; ia<16; ia++){
		DepArray1[ia][0] = IndArray1[ia][0] * (1.0 - 0.5 * (IndArray2[ia][0] + 
			IndArray3[ia][0]) + IndArray2[ia][0] * IndArray3[ia][0]/3.0) * 
			(1.0 - NonAIDSmortProb[ia][SexInd]);
		DepArray2[ia][0] = IndArray2[ia][0] * (1.0 - 0.5 * (IndArray1[ia][0] + 
			IndArray3[ia][0]) + IndArray1[ia][0] * IndArray3[ia][0]/3.0) * 
			(1.0 - NonAIDSmortProb[ia][SexInd]);
		DepArray3[ia][0] = IndArray3[ia][0] * (1.0 - 0.5 * (IndArray1[ia][0] + 
			IndArray2[ia][0]) + IndArray1[ia][0] * IndArray2[ia][0]/3.0) * 
			(1.0 - NonAIDSmortProb[ia][SexInd]);
		Remain[ia][0] = 1.0 - NonAIDSmortProb[ia][SexInd] - DepArray1[ia][0] - 
			DepArray2[ia][0] - DepArray3[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				DepArray1[ia][is] = IndArray1[ia][is] * (1.0 - 0.5 * (IndArray2[ia][is] + 
					IndArray3[ia][is]) + IndArray2[ia][is] * IndArray3[ia][is]/3.0) * 
					(1.0 - NonAIDSmortProb[ia][SexInd]);
				DepArray2[ia][is] = IndArray2[ia][is] * (1.0 - 0.5 * (IndArray1[ia][is] + 
					IndArray3[ia][is]) + IndArray1[ia][is] * IndArray3[ia][is]/3.0) * 
					(1.0 - NonAIDSmortProb[ia][SexInd]);
				DepArray3[ia][is] = IndArray3[ia][is] * (1.0 - 0.5 * (IndArray1[ia][is] + 
					IndArray2[ia][is]) + IndArray1[ia][is] * IndArray2[ia][is]/3.0) * 
					(1.0 - NonAIDSmortProb[ia][SexInd]);
				Remain[ia][is] = 1.0 - NonAIDSmortProb[ia][SexInd] - DepArray1[ia][is] - 
					DepArray2[ia][is] - DepArray3[ia][is];
			}
		}
	}
}

void SexuallyExp::ConvertToDependent4(double IndArray1[16][6], double IndArray2[16][6],
		double IndArray3[16][6], double IndArray4[16][6], double DepArray1[16][6], 
		double DepArray2[16][6], double DepArray3[16][6], double DepArray4[16][6])
{
	int ia, is;

	for(ia=0; ia<16; ia++){
		DepArray1[ia][0] = IndArray1[ia][0] * (1.0 - 0.5 * (IndArray2[ia][0] + 
			IndArray3[ia][0] + IndArray4[ia][0]) + (IndArray2[ia][0] * IndArray3[ia][0] +
			IndArray2[ia][0] * IndArray4[ia][0] + IndArray3[ia][0] * IndArray4[ia][0])/3.0 -
			0.25 * IndArray2[ia][0] * IndArray3[ia][0] * IndArray4[ia][0]) * 
			(1.0 - NonAIDSmortProb[ia][SexInd]);
		DepArray2[ia][0] = IndArray2[ia][0] * (1.0 - 0.5 * (IndArray1[ia][0] + 
			IndArray3[ia][0] + IndArray4[ia][0]) + (IndArray1[ia][0] * IndArray3[ia][0] +
			IndArray1[ia][0] * IndArray4[ia][0] + IndArray3[ia][0] * IndArray4[ia][0])/3.0 -
			0.25 * IndArray1[ia][0] * IndArray3[ia][0] * IndArray4[ia][0]) * 
			(1.0 - NonAIDSmortProb[ia][SexInd]);
		DepArray3[ia][0] = IndArray3[ia][0] * (1.0 - 0.5 * (IndArray1[ia][0] + 
			IndArray2[ia][0] + IndArray4[ia][0]) + (IndArray1[ia][0] * IndArray2[ia][0] +
			IndArray1[ia][0] * IndArray4[ia][0] + IndArray2[ia][0] * IndArray4[ia][0])/3.0 -
			0.25 * IndArray1[ia][0] * IndArray2[ia][0] * IndArray4[ia][0]) * 
			(1.0 - NonAIDSmortProb[ia][SexInd]);
		DepArray4[ia][0] = IndArray4[ia][0] * (1.0 - 0.5 * (IndArray1[ia][0] + 
			IndArray2[ia][0] + IndArray3[ia][0]) + (IndArray1[ia][0] * IndArray2[ia][0] +
			IndArray1[ia][0] * IndArray3[ia][0] + IndArray2[ia][0] * IndArray3[ia][0])/3.0 -
			0.25 * IndArray1[ia][0] * IndArray2[ia][0] * IndArray3[ia][0]) * 
			(1.0 - NonAIDSmortProb[ia][SexInd]);
		Remain[ia][0] = 1.0 - NonAIDSmortProb[ia][SexInd] - DepArray1[ia][0] - 
			DepArray2[ia][0] - DepArray3[ia][0] - DepArray4[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				DepArray1[ia][is] = IndArray1[ia][is] * (1.0 - 0.5 * (IndArray2[ia][is] + 
					IndArray3[ia][is] + IndArray4[ia][is]) + (IndArray2[ia][is] * IndArray3[ia][is] +
					IndArray2[ia][is] * IndArray4[ia][is] + IndArray3[ia][is] * IndArray4[ia][is])/3.0 -
					0.25 * IndArray2[ia][is] * IndArray3[ia][is] * IndArray4[ia][is]) * 
					(1.0 - NonAIDSmortProb[ia][SexInd]);
				DepArray2[ia][is] = IndArray2[ia][is] * (1.0 - 0.5 * (IndArray1[ia][is] + 
					IndArray3[ia][is] + IndArray4[ia][is]) + (IndArray1[ia][is] * IndArray3[ia][is] +
					IndArray1[ia][is] * IndArray4[ia][is] + IndArray3[ia][is] * IndArray4[ia][is])/3.0 -
					0.25 * IndArray1[ia][is] * IndArray3[ia][is] * IndArray4[ia][is]) * 
					(1.0 - NonAIDSmortProb[ia][SexInd]);
				DepArray3[ia][is] = IndArray3[ia][is] * (1.0 - 0.5 * (IndArray1[ia][is] + 
					IndArray2[ia][is] + IndArray4[ia][is]) + (IndArray1[ia][is] * IndArray2[ia][is] +
					IndArray1[ia][is] * IndArray4[ia][is] + IndArray2[ia][is] * IndArray4[ia][is])/3.0 -
					0.25 * IndArray1[ia][is] * IndArray2[ia][is] * IndArray4[ia][is]) * 
					(1.0 - NonAIDSmortProb[ia][SexInd]);
				DepArray4[ia][is] = IndArray4[ia][is] * (1.0 - 0.5 * (IndArray1[ia][is] + 
					IndArray2[ia][is] + IndArray3[ia][is]) + (IndArray1[ia][is] * IndArray2[ia][is] +
					IndArray1[ia][is] * IndArray3[ia][is] + IndArray2[ia][is] * IndArray3[ia][is])/3.0 -
					0.25 * IndArray1[ia][is] * IndArray2[ia][is] * IndArray3[ia][is]) * 
					(1.0 - NonAIDSmortProb[ia][SexInd]);
				Remain[ia][is] = 1.0 - NonAIDSmortProb[ia][SexInd] - DepArray1[ia][is] - 
					DepArray2[ia][is] - DepArray3[ia][is] - DepArray4[ia][is];
			}
		}
	}
}

void SexuallyExp::CalcAgeChanges(NonHIV* m)
{
	int ia, is, id;
	double ExitRates[16][6];

	for(ia=0; ia<16; ia++){
		for(is=0; is<6; is++){
			if(SexInd==0 && VirginInd==0){ExitRates[ia][is] = AgeExitRateM[ia][is];}
			if(SexInd==1 && VirginInd==0){ExitRates[ia][is] = AgeExitRateF[ia][is];}
		}
		if(SexInd==0 && VirginInd==1){ExitRates[ia][0] = VirginAgeExitRate[ia][0];}
		if(SexInd==1 && VirginInd==1){ExitRates[ia][0] = VirginAgeExitRate[ia][1];}
	}

	for(ia=0; ia<15; ia++){
		for(id=0; id<m->nStates; id++){
			m->NumberByStage0[15-ia][id] = m->NumberByStage0[15-ia][id] * (1.0 - 
				ExitRates[15-ia][0]) + m->NumberByStage0[14-ia][id] * ExitRates[14-ia][0];
			if(HIVind==1 && VirginInd==0){
				m->NumberByStage1[15-ia][id] = m->NumberByStage1[15-ia][id] * (1.0 - 
					ExitRates[15-ia][1]) + m->NumberByStage1[14-ia][id] * ExitRates[14-ia][1];
				m->NumberByStage2[15-ia][id] = m->NumberByStage2[15-ia][id] * (1.0 - 
					ExitRates[15-ia][2]) + m->NumberByStage2[14-ia][id] * ExitRates[14-ia][2];
				m->NumberByStage3[15-ia][id] = m->NumberByStage3[15-ia][id] * (1.0 - 
					ExitRates[15-ia][3]) + m->NumberByStage3[14-ia][id] * ExitRates[14-ia][3];
				m->NumberByStage4[15-ia][id] = m->NumberByStage4[15-ia][id] * (1.0 - 
					ExitRates[15-ia][4]) + m->NumberByStage4[14-ia][id] * ExitRates[14-ia][4];
				m->NumberByStage5[15-ia][id] = m->NumberByStage5[15-ia][id] * (1.0 - 
					ExitRates[15-ia][5]) + m->NumberByStage5[14-ia][id] * ExitRates[14-ia][5];
			}
		}
	}
	// Repeat the same code for the first age category (only age exits permitted).
	for(id=0; id<m->nStates; id++){
		m->NumberByStage0[0][id] = m->NumberByStage0[0][id] * (1.0 - ExitRates[0][0]);
		if(HIVind==1 && VirginInd==0){
			m->NumberByStage1[0][id] = m->NumberByStage1[0][id] * (1.0 - ExitRates[0][1]);
			m->NumberByStage2[0][id] = m->NumberByStage2[0][id] * (1.0 - ExitRates[0][2]);
			m->NumberByStage3[0][id] = m->NumberByStage3[0][id] * (1.0 - ExitRates[0][3]);
			m->NumberByStage4[0][id] = m->NumberByStage4[0][id] * (1.0 - ExitRates[0][4]);
			m->NumberByStage5[0][id] = m->NumberByStage5[0][id] * (1.0 - ExitRates[0][5]);
		}
	}
}

double SexuallyExp::ReturnHIVprev()
{
	int ia;
	double numerator, denominator, HIVprev;

	numerator = 0;
	denominator = 0;
	for(ia=0; ia<16; ia++){
		numerator += NumbersByHIVstage[ia][0];
		denominator += TotalAlive[ia];
	}
	HIVprev = 1.0 - numerator/denominator;

	return HIVprev; 
}

double SexuallyExp::ReturnTotalGUD()
{
	int ia, is;
	double TotalGUD;

	TotalGUD = 0;
	for(ia=0; ia<16; ia++){
		TotalGUD += NumbersByHIVstage[ia][0] * GUDpropn[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				TotalGUD += NumbersByHIVstage[ia][is] * GUDpropn[ia][is];}
		}
	}

	return TotalGUD;
}

SexuallyExpM::SexuallyExpM()
{
	SexInd = 0;
}

void SexuallyExpM::Reset()
{
	int ia, is, iz;

	for(ia=0; ia<16; ia++){
		NumbersByHIVstage[ia][0] = TotalAlive[ia];
		if(HIVind==1){
			for(is=1; is<6; is++){
				NumbersByHIVstage[ia][is] = 0;}
		}
	}

	if(HIVind==1 && RiskGroup==1){
		for(ia=1; ia<8; ia++){
			NumbersByHIVstage[ia][0] = TotalAlive[ia] * (1.0 - InitHIVprevHigh);
			NumbersByHIVstage[ia][2] = TotalAlive[ia] * InitHIVprevHigh;
			for(iz=0; iz<7; iz++){
				MHSV.PropnByStage2[ia][iz] = MHSV.PropnByStage0[ia][iz];
				MTP.PropnByStage2[ia][iz] = MTP.PropnByStage0[ia][iz];
				MHD.PropnByStage2[ia][iz] = MHD.PropnByStage0[ia][iz];
				MNG.PropnByStage2[ia][iz] = MNG.PropnByStage0[ia][iz];
				MCT.PropnByStage2[ia][iz] = MCT.PropnByStage0[ia][iz];
				MTV.PropnByStage2[ia][iz] = MTV.PropnByStage0[ia][iz];
			}
		}
	}
}

void SexuallyExpM::GetAllNumbersBySTDstage()
{
	if(TPind==1){GetNumbersBySTDstage(&MTP);}
	if(HSVind==1){GetNumbersBySTDstage(&MHSV);}
	if(HDind==1){GetNumbersBySTDstage(&MHD);}
	if(NGind==1){GetNumbersBySTDstage(&MNG);}
	if(CTind==1){GetNumbersBySTDstage(&MCT);}
	if(TVind==1){GetNumbersBySTDstage(&MTV);}
}

void SexuallyExpM::GetAllPropnsBySTDstage()
{
	if(TPind==1){GetPropnsBySTDstage(&MTP);}
	if(HSVind==1){GetPropnsBySTDstage(&MHSV);}
	if(HDind==1){GetPropnsBySTDstage(&MHD);}
	if(NGind==1){GetPropnsBySTDstage(&MNG);}
	if(CTind==1){GetPropnsBySTDstage(&MCT);}
	if(TVind==1){GetPropnsBySTDstage(&MTV);}
}

void SexuallyExpM::UpdateSyndromePropns()
{
	int ia;

	for(ia=0; ia<16; ia++){
		GUDpropn[ia][0] = 1.0 - (1.0 - MHSV.PropnByStage0[ia][1] - MHSV.PropnByStage0[ia][3])
			* (1 - MTP.PropnByStage0[ia][2]) * (1.0 - MHD.PropnByStage0[ia][1]);
		GUDpropn[ia][1] = 1.0 - (1.0 - MHSV.PropnByStage1[ia][1] - MHSV.PropnByStage1[ia][3])
			* (1 - MTP.PropnByStage1[ia][2]) * (1.0 - MHD.PropnByStage1[ia][1]);
		GUDpropn[ia][2] = 1.0 - (1.0 - MHSV.PropnByStage2[ia][1] - MHSV.PropnByStage2[ia][3])
			* (1 - MTP.PropnByStage2[ia][2]) * (1.0 - MHD.PropnByStage2[ia][1]);
		GUDpropn[ia][3] = 1.0 - (1.0 - MHSV.PropnByStage3[ia][1] - MHSV.PropnByStage3[ia][3])
			* (1 - MTP.PropnByStage3[ia][2]) * (1.0 - MHD.PropnByStage3[ia][1]);
		GUDpropn[ia][4] = 1.0 - (1.0 - MHSV.PropnByStage4[ia][1] - MHSV.PropnByStage4[ia][3])
			* (1 - MTP.PropnByStage4[ia][2]) * (1.0 - MHD.PropnByStage4[ia][1]);
		GUDpropn[ia][5] = 1.0 - (1.0 - MHSV.PropnByStage5[ia][1] - MHSV.PropnByStage5[ia][3])
			* (1 - MTP.PropnByStage5[ia][2]) * (1.0 - MHD.PropnByStage5[ia][1]);
		DischargePropn[ia][0] = (1 - GUDpropn[ia][0])*(1.0 - (1.0 - MNG.PropnByStage0[ia][1])
			* (1.0 - MCT.PropnByStage0[ia][1]) * (1.0 - MTV.PropnByStage0[ia][1]));
		DischargePropn[ia][1] = (1 - GUDpropn[ia][1])*(1.0 - (1.0 - MNG.PropnByStage1[ia][1])
			* (1.0 - MCT.PropnByStage1[ia][1]) * (1.0 - MTV.PropnByStage1[ia][1]));
		DischargePropn[ia][2] = (1 - GUDpropn[ia][2])*(1.0 - (1.0 - MNG.PropnByStage2[ia][1])
			* (1.0 - MCT.PropnByStage2[ia][1]) * (1.0 - MTV.PropnByStage2[ia][1]));
		DischargePropn[ia][3] = (1 - GUDpropn[ia][3])*(1.0 - (1.0 - MNG.PropnByStage3[ia][1])
			* (1.0 - MCT.PropnByStage3[ia][1]) * (1.0 - MTV.PropnByStage3[ia][1]));
		DischargePropn[ia][4] = (1 - GUDpropn[ia][4])*(1.0 - (1.0 - MNG.PropnByStage4[ia][1])
			* (1.0 - MCT.PropnByStage4[ia][1]) * (1.0 - MTV.PropnByStage4[ia][1]));
		DischargePropn[ia][5] = (1 - GUDpropn[ia][5])*(1.0 - (1.0 - MNG.PropnByStage5[ia][1])
			* (1.0 - MCT.PropnByStage5[ia][1]) * (1.0 - MTV.PropnByStage5[ia][1]));
		AsympSTDpropn[ia][0] = (1.0 - (MHSV.PropnByStage0[ia][0] + MHSV.PropnByStage0[ia][2] 
			+ MHSV.PropnByStage0[ia][4])*(MTP.PropnByStage0[ia][0] + MTP.PropnByStage0[ia][4]
			+ MTP.PropnByStage0[ia][5] + MTP.PropnByStage0[ia][6]) * MHD.PropnByStage0[ia][0]
			* MNG.PropnByStage0[ia][0] * MCT.PropnByStage0[ia][0] * MTV.PropnByStage0[ia][0])
			- GUDpropn[ia][0] - DischargePropn[ia][0];
		AsympSTDpropn[ia][1] = (1.0 - (MHSV.PropnByStage1[ia][0] + MHSV.PropnByStage1[ia][2] 
			+ MHSV.PropnByStage1[ia][4])*(MTP.PropnByStage1[ia][0] + MTP.PropnByStage1[ia][4]
			+ MTP.PropnByStage1[ia][5] + MTP.PropnByStage1[ia][6]) * MHD.PropnByStage1[ia][0]
			* MNG.PropnByStage1[ia][0] * MCT.PropnByStage1[ia][0] * MTV.PropnByStage1[ia][0])
			- GUDpropn[ia][1] - DischargePropn[ia][1];
		AsympSTDpropn[ia][2] = (1.0 - (MHSV.PropnByStage2[ia][0] + MHSV.PropnByStage2[ia][2] 
			+ MHSV.PropnByStage2[ia][4])*(MTP.PropnByStage2[ia][0] + MTP.PropnByStage2[ia][4]
			+ MTP.PropnByStage2[ia][5] + MTP.PropnByStage2[ia][6]) * MHD.PropnByStage2[ia][0]
			* MNG.PropnByStage2[ia][0] * MCT.PropnByStage2[ia][0] * MTV.PropnByStage2[ia][0])
			- GUDpropn[ia][2] - DischargePropn[ia][2];
		AsympSTDpropn[ia][3] = (1.0 - (MHSV.PropnByStage3[ia][0] + MHSV.PropnByStage3[ia][2] 
			+ MHSV.PropnByStage3[ia][4])*(MTP.PropnByStage3[ia][0] + MTP.PropnByStage3[ia][4]
			+ MTP.PropnByStage3[ia][5] + MTP.PropnByStage3[ia][6]) * MHD.PropnByStage3[ia][0]
			* MNG.PropnByStage3[ia][0] * MCT.PropnByStage3[ia][0] * MTV.PropnByStage3[ia][0])
			- GUDpropn[ia][3] - DischargePropn[ia][3];
		AsympSTDpropn[ia][4] = (1.0 - (MHSV.PropnByStage4[ia][0] + MHSV.PropnByStage4[ia][2] 
			+ MHSV.PropnByStage4[ia][4])*(MTP.PropnByStage4[ia][0] + MTP.PropnByStage4[ia][4]
			+ MTP.PropnByStage4[ia][5] + MTP.PropnByStage4[ia][6]) * MHD.PropnByStage4[ia][0]
			* MNG.PropnByStage4[ia][0] * MCT.PropnByStage4[ia][0] * MTV.PropnByStage4[ia][0])
			- GUDpropn[ia][4] - DischargePropn[ia][4];
		AsympSTDpropn[ia][5] = (1.0 - (MHSV.PropnByStage5[ia][0] + MHSV.PropnByStage5[ia][2] 
			+ MHSV.PropnByStage5[ia][4])*(MTP.PropnByStage5[ia][0] + MTP.PropnByStage5[ia][4]
			+ MTP.PropnByStage5[ia][5] + MTP.PropnByStage5[ia][6]) * MHD.PropnByStage5[ia][0]
			* MNG.PropnByStage5[ia][0] * MCT.PropnByStage5[ia][0] * MTV.PropnByStage5[ia][0])
			- GUDpropn[ia][5] - DischargePropn[ia][5];
	}
}

void SexuallyExpM::GetAllSTDcofactors()
{
	int ia, is;

	for(ia=0; ia<16; ia++){
		for(is=0; is<6; is++){
			STDcofactor[ia][is] = 1.0;} // Default (no STD cofactors)
	}

	if(CofactorType==1){ // Multiplicative cofactors
		if(TPind==1){GetSTDcofactor(&MTP, &TPtransitionM);}
		if(HSVind==1){GetSTDcofactor(&MHSV, &HSVtransitionM);}
		if(HDind==1){GetSTDcofactor(&MHD, &HDtransitionM);}
		if(NGind==1){GetSTDcofactor(&MNG, &NGtransitionM);}
		if(CTind==1){GetSTDcofactor(&MCT, &CTtransitionM);}
		if(TVind==1){GetSTDcofactor(&MTV, &TVtransitionM);}
	}

	else if(CofactorType==2){ // Saturation cofactors
		for(ia=0; ia<16; ia++){
			STDcofactor[ia][0] += GUDpropn[ia][0] * SuscepIncreaseSyndrome[0][0] +
				DischargePropn[ia][0] * SuscepIncreaseSyndrome[1][0] +
				AsympSTDpropn[ia][0] * SuscepIncreaseSyndrome[2][0];
			for(is=1; is<6; is++){
				STDcofactor[ia][is] += GUDpropn[ia][is] * InfecIncreaseSyndrome[0][0] +
					DischargePropn[ia][is] * InfecIncreaseSyndrome[1][0] +
					AsympSTDpropn[ia][is] * InfecIncreaseSyndrome[2][0];}
		}
	}
}

void SexuallyExpM::CalcTransmissionProb()
{
	int ia, is;
	double sumx;

	// Calculate prob of transmitting STD in single act of unprotected sex, weighted by
	// number of individuals with STD (cols HT to IB in each risk group sheet)

	if(TPind==1){MTP.CalcTransmissionProb(&TPtransitionM);}
	if(HSVind==1){MHSV.CalcTransmissionProb(&HSVtransitionM);}
	if(HDind==1){MHD.CalcTransmissionProb(&HDtransitionM);}
	if(NGind==1){MNG.CalcTransmissionProb(&NGtransitionM);}
	if(CTind==1){MCT.CalcTransmissionProb(&CTtransitionM);}
	if(TVind==1){MTV.CalcTransmissionProb(&TVtransitionM);}

	if(HIVind==1){
		for(ia=0; ia<16; ia++){
			sumx = 0;
			for(is=1; is<6; is++){
				sumx += NumbersByHIVstage[ia][is] * (1.0 + 
					HIVtransitionM.HIVinfecIncrease[is-1]) * STDcofactor[ia][is];}
			// FSW-client transmission prob
			if(RiskGroup==1){
				WeightedProbTransm[ia][0] = sumx * HIVtransitionM.TransmProb[0] *
					RatioAsympToAveM;}
			// ST relationship transmission prob
			if(NumberPartnersS1>0 && RiskGroup==1){
				WeightedProbTransm[ia][1] = sumx * HIVtransitionM.TransmProb[1] *
					RatioAsympToAveM;}
			if(NumberPartnersS2>0 && RiskGroup==2){
				WeightedProbTransm[ia][3] = sumx * HIVtransitionM.TransmProb[3] *
					RatioAsympToAveM;}
			if((NumberPartnersS1>0 && RiskGroup==2) || (NumberPartnersS2>0 && RiskGroup==1)){
				WeightedProbTransm[ia][2] = sumx * HIVtransitionM.TransmProb[2] *
					RatioAsympToAveM;}
			// LT relationship transmission prob
			if(NumberPartnersL1>0 && RiskGroup==1){
				WeightedProbTransm[ia][4] = sumx * HIVtransitionM.TransmProb[4] *
					RatioAsympToAveM;}
			if(NumberPartnersL2>0 && RiskGroup==2){
				WeightedProbTransm[ia][6] = sumx * HIVtransitionM.TransmProb[6] *
					RatioAsympToAveM;}
			if((NumberPartnersL1>0 && RiskGroup==2) || (NumberPartnersL2>0 && RiskGroup==1)){
				WeightedProbTransm[ia][5] = sumx * HIVtransitionM.TransmProb[5] *
					RatioAsympToAveM;}
		}
	}

	// Calculate AnnFSWcontacts

	if(RiskGroup==1){
		AnnFSWcontacts = 0;
		for(ia=0; ia<16; ia++){
			AnnFSWcontacts += TotalAlive[ia] * AgeEffectFSWcontact[ia];}
		AnnFSWcontacts *= FSWcontactConstant * PartnerEffectFSWcontact[FSWcontactOffset];
	}

	// Sum weighted probabilities across risk groups (1st set of tables in 'STD transmission'
	// sheet).

	if(TPind==1){CalcOneTransmProb(&TPtransitionM, &MTP);}
	if(HSVind==1){CalcOneTransmProb(&HSVtransitionM, &MHSV);}
	if(HDind==1){CalcOneTransmProb(&HDtransitionM, &MHD);}
	if(NGind==1){CalcOneTransmProb(&NGtransitionM, &MNG);}
	if(CTind==1){CalcOneTransmProb(&CTtransitionM, &MCT);}
	if(TVind==1){CalcOneTransmProb(&TVtransitionM, &MTV);}
	if(HIVind==1){CalcHIVtransmProb(&HIVtransitionM);}
}

void SexuallyExpM::CalcOneTransmProb(STDtransition* a, TradnalSTD* b)
{
	int ia;

	if(RiskGroup==1){
		b->WeightedProbTransmFSW = 0;
		for(ia=0; ia<16; ia++){
			a->TransmProbS1to1[ia] += b->WeightedProbTransm[ia] * NumberPartnersS1;
			a->TransmProbS1to2[ia] += b->WeightedProbTransm[ia] * NumberPartnersS2;
			a->TransmProbL1to1[ia] += b->WeightedProbTransm[ia] * NumberPartnersL1;
			a->TransmProbL1to2[ia] += b->WeightedProbTransm[ia] * NumberPartnersL2;
			b->WeightedProbTransmFSW += b->WeightedProbTransm[ia] * AgeEffectFSWcontact[ia];
		}
		b->WeightedProbTransmFSW *= PartnerEffectFSWcontact[FSWcontactOffset] *
			FSWcontactConstant;
		a->InfectProbFSW[0] += b->WeightedProbTransmFSW;
	}
	else{
		for(ia=0; ia<16; ia++){
			a->TransmProbS2to1[ia] += b->WeightedProbTransm[ia] * NumberPartnersS1;
			a->TransmProbS2to2[ia] += b->WeightedProbTransm[ia] * NumberPartnersS2;
			a->TransmProbL2to1[ia] += b->WeightedProbTransm[ia] * NumberPartnersL1;
			a->TransmProbL2to2[ia] += b->WeightedProbTransm[ia] * NumberPartnersL2;
		}
	}
}

void SexuallyExpM::CalcInfectionProb()
{
	if(HIVind==1){CalcHIVinfectProb(&HIVtransitionF);}
	if(HSVind==1){CalcHSVinfectProb(&HSVtransitionF, &MHSV);}
	if(TPind==1){CalcOneInfectProb(&TPtransitionF, &MTP);}
	if(HDind==1){CalcOneInfectProb(&HDtransitionF, &MHD);}
	if(NGind==1){CalcOneInfectProb(&NGtransitionF, &MNG);}
	if(CTind==1){CalcOneInfectProb(&CTtransitionF, &MCT);}
	if(TVind==1){CalcOneInfectProb(&TVtransitionF, &MTV);}
}

void SexuallyExpM::CalcOneInfectProb(STDtransition* a, TradnalSTD* b)
{
	int ia;
	double FSWcontactRate;

	if(RiskGroup==1){
		// Calculate FSWcontactRate
		FSWcontactRate = FSWcontactConstant;
		if(NumberPartners==0){FSWcontactRate *= PartnerEffectFSWcontact[0];}
		if(NumberPartners==1){
			if(NumberPartnersS1 + NumberPartnersS2==1){
				FSWcontactRate *= PartnerEffectFSWcontact[1];}
			else{
				FSWcontactRate *= PartnerEffectFSWcontact[2];}
		}
		if(NumberPartners==2){
			if(NumberPartnersS1 + NumberPartnersS2==2){
				FSWcontactRate *= PartnerEffectFSWcontact[3];}
			else{
				FSWcontactRate *= PartnerEffectFSWcontact[4];}
		}
		// Calculate InfectProb
		for(ia=0; ia<16; ia++){
			b->InfectProb[ia] = 1.0;
			if(NumberPartnersS1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS1from1[ia], NumberPartnersS1 * 
					FreqSexST[ia][0] * CycleS/CycleD);}
			if(NumberPartnersS2>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS1from2[ia], NumberPartnersS2 * 
					FreqSexST[ia][0] * CycleS/CycleD);}
			if(NumberPartnersL1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL1from1[ia], 
					FreqSexLT[ia][0] * CycleS/CycleD);}
			if(NumberPartnersL2>0 && NoBacterialTransm12==0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL1from2[ia], 
					FreqSexLT[ia][0] * CycleS/CycleD);}
			b->InfectProb[ia] *= pow(1.0 - a->InfectProbFSW[ia], FSWcontactRate * 
				AgeEffectFSWcontact[ia]/CycleD);
			b->InfectProb[ia] = 1.0 - b->InfectProb[ia];
		}
	}
	else{ // Calculate InfectProb for low risk group
		for(ia=0; ia<16; ia++){
			b->InfectProb[ia] = 1.0;
			if(NumberPartnersS1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS2from1[ia], 
					FreqSexST[ia][0] * CycleS/CycleD);}
			if(NumberPartnersS2>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS2from2[ia], 
					FreqSexST[ia][0] * CycleS/CycleD);}
			if(NumberPartnersL1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL2from1[ia], 
					FreqSexLT[ia][0] * CycleS/CycleD);}
			if(NumberPartnersL2>0 && NoBacterialTransm22==0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL2from2[ia], 
					FreqSexLT[ia][0] * CycleS/CycleD);}
			b->InfectProb[ia] = 1.0 - b->InfectProb[ia];
		}
	}
}

void SexuallyExpM::CalcHSVinfectProb(HerpesTransition* a, Herpes* b)
{
	// The same as the CalcOneInfectProb function except that the arguments are different
	// and we use NoViralTransm in place of NoBacterialTransm.

	int ia;
	double FSWcontactRate;

	if(RiskGroup==1){
		// Calculate FSWcontactRate
		FSWcontactRate = FSWcontactConstant;
		if(NumberPartners==0){FSWcontactRate *= PartnerEffectFSWcontact[0];}
		if(NumberPartners==1){
			if(NumberPartnersS1 + NumberPartnersS2==1){
				FSWcontactRate *= PartnerEffectFSWcontact[1];}
			else{
				FSWcontactRate *= PartnerEffectFSWcontact[2];}
		}
		if(NumberPartners==2){
			if(NumberPartnersS1 + NumberPartnersS2==2){
				FSWcontactRate *= PartnerEffectFSWcontact[3];}
			else{
				FSWcontactRate *= PartnerEffectFSWcontact[4];}
		}
		// Calculate InfectProb
		for(ia=0; ia<16; ia++){
			b->InfectProb[ia] = 1.0;
			if(NumberPartnersS1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS1from1[ia], NumberPartnersS1 * 
					FreqSexST[ia][0] * CycleS/CycleD);}
			if(NumberPartnersS2>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS1from2[ia], NumberPartnersS2 * 
					FreqSexST[ia][0] * CycleS/CycleD);}
			if(NumberPartnersL1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL1from1[ia], 
					FreqSexLT[ia][0] * CycleS/CycleD);}
			if(NumberPartnersL2>0 && NoViralTransm12==0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL1from2[ia], 
					FreqSexLT[ia][0] * CycleS/CycleD);}
			b->InfectProb[ia] *= pow(1.0 - a->InfectProbFSW[ia], FSWcontactRate * 
				AgeEffectFSWcontact[ia]/CycleD);
			b->InfectProb[ia] = 1.0 - b->InfectProb[ia];
		}
	}
	else{ // Calculate InfectProb for low risk group
		for(ia=0; ia<16; ia++){
			b->InfectProb[ia] = 1.0;
			if(NumberPartnersS1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS2from1[ia], 
					FreqSexST[ia][0] * CycleS/CycleD);}
			if(NumberPartnersS2>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS2from2[ia], 
					FreqSexST[ia][0] * CycleS/CycleD);}
			if(NumberPartnersL1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL2from1[ia], 
					FreqSexLT[ia][0] * CycleS/CycleD);}
			if(NumberPartnersL2>0 && NoViralTransm22==0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL2from2[ia], 
					FreqSexLT[ia][0] * CycleS/CycleD);}
			b->InfectProb[ia] = 1.0 - b->InfectProb[ia];
		}
	}
}

void SexuallyExpM::CalcHIVinfectProb(HIVtransition* a)
{
	// Similar to CalcHSVinfectProb function except that the arguments are different,
	// we allow for STD cofactors, we use HIVinfectProb instead of b->InfectProb, and we
	// set a max on the probability per contact (because of the STD cofactors).

	int ia;
	double FSWcontactRate, ProbPerAct;

	if(RiskGroup==1){
		// Calculate FSWcontactRate
		FSWcontactRate = FSWcontactConstant;
		if(NumberPartners==0){FSWcontactRate *= PartnerEffectFSWcontact[0];}
		if(NumberPartners==1){
			if(NumberPartnersS1 + NumberPartnersS2==1){
				FSWcontactRate *= PartnerEffectFSWcontact[1];}
			else{
				FSWcontactRate *= PartnerEffectFSWcontact[2];}
		}
		if(NumberPartners==2){
			if(NumberPartnersS1 + NumberPartnersS2==2){
				FSWcontactRate *= PartnerEffectFSWcontact[3];}
			else{
				FSWcontactRate *= PartnerEffectFSWcontact[4];}
		}
		// Calculate InfectProb
		for(ia=0; ia<16; ia++){
			HIVinfectProb[ia] = 1.0;
			//HIVinfectProbST[ia] = 1.0;
			//HIVinfectProbLT[ia] = 1.0;
			//HIVinfectProbCS[ia] = 1.0;
			if(NumberPartnersS1>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbS1from1[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, NumberPartnersS1 * 
					FreqSexST[ia][0] * CycleS/CycleD);
				//HIVinfectProbST[ia] *= pow(1.0 - ProbPerAct, NumberPartnersS1 * 
				//	FreqSexST[ia][0] * CycleS/CycleD);
			}
			if(NumberPartnersS2>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbS1from2[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, NumberPartnersS2 * 
					FreqSexST[ia][0] * CycleS/CycleD);
				//HIVinfectProbST[ia] *= pow(1.0 - ProbPerAct, NumberPartnersS2 * 
				//	FreqSexST[ia][0] * CycleS/CycleD);
			}
			if(NumberPartnersL1>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbL1from1[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][0] * CycleS/CycleD);
				//HIVinfectProbLT[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][0] * CycleS/CycleD);
			}
			if(NumberPartnersL2>0 && NoViralTransm12==0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbL1from2[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][0] * CycleS/CycleD);
				//HIVinfectProbLT[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][0] * CycleS/CycleD);
			}
			ProbPerAct = STDcofactor[ia][0] * a->InfectProbFSW[ia];
			if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
			HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FSWcontactRate * 
				AgeEffectFSWcontact[ia]/CycleD);
			//HIVinfectProbCS[ia] *= pow(1.0 - ProbPerAct, FSWcontactRate * 
			//	AgeEffectFSWcontact[ia]/CycleD);
			HIVinfectProb[ia] = 1.0 - HIVinfectProb[ia];
			//HIVinfectProbST[ia] = 1.0 - HIVinfectProbST[ia];
			//HIVinfectProbLT[ia] = 1.0 - HIVinfectProbLT[ia];
			//HIVinfectProbCS[ia] = 1.0 - HIVinfectProbCS[ia];
		}
	}
	else{ // Calculate InfectProb for low risk group
		for(ia=0; ia<16; ia++){
			HIVinfectProb[ia] = 1.0;
			//HIVinfectProbST[ia] = 1.0;
			//HIVinfectProbLT[ia] = 1.0;
			if(NumberPartnersS1>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbS2from1[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexST[ia][0] * CycleS/CycleD);
				//HIVinfectProbST[ia] *= pow(1.0 - ProbPerAct, FreqSexST[ia][0] * CycleS/CycleD);
			}
			if(NumberPartnersS2>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbS2from2[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexST[ia][0] * CycleS/CycleD);
				//HIVinfectProbST[ia] *= pow(1.0 - ProbPerAct, FreqSexST[ia][0] * CycleS/CycleD);
			}
			if(NumberPartnersL1>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbL2from1[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][0] * CycleS/CycleD);
				//HIVinfectProbLT[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][0] * CycleS/CycleD);
			}
			if(NumberPartnersL2>0 && NoViralTransm12==0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbL2from2[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][0] * CycleS/CycleD);
				//HIVinfectProbLT[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][0] * CycleS/CycleD);
			}
			HIVinfectProb[ia] = 1.0 - HIVinfectProb[ia];
			//HIVinfectProbST[ia] = 1.0 - HIVinfectProbST[ia];
			//HIVinfectProbLT[ia] = 1.0 - HIVinfectProbLT[ia];
		}
	}
}

void SexuallyExpM::CalcSTDtransitions()
{
	if(HSVind==1){MHSV.CalcSTDtransitions(&HSVtransitionM, 0);}
	if(TPind==1){MTP.CalcSTDtransitions(&TPtransitionM, 0);}
	if(HDind==1){MHD.CalcSTDtransitions(&HDtransitionM, 0);}
	if(NGind==1){MNG.CalcSTDtransitions(&NGtransitionM, 0);}
	if(CTind==1){MCT.CalcSTDtransitions(&CTtransitionM, 0);}
	if(TVind==1){MTV.CalcSTDtransitions(&TVtransitionM, 0);}
	if(HIVind==1){CalcHIVtransitions(&HIVtransitionM);}
}

void SexuallyExpM::AllHIVstageChanges()
{
	if(HSVind==1){HIVstageChanges(&MHSV);}
	if(TPind==1){HIVstageChanges(&MTP);}
	if(HDind==1){HIVstageChanges(&MHD);}
	if(NGind==1){HIVstageChanges(&MNG);}
	if(CTind==1){HIVstageChanges(&MCT);}
	if(TVind==1){HIVstageChanges(&MTV);}
}

void SexuallyExpM::GetNumbersRemaining1(NonHIV* a)
{
	int ia, iz;

	for(ia=0; ia<16; ia++){
		for(iz=0; iz<a->nStates; iz++){
			a->NumberByStage0[ia][iz] = TempNumbersByHIVstage[ia][0] * 
				a->PropnByStage0[ia][iz];
			if(HIVind==1){
				a->NumberByStage1[ia][iz] = TempNumbersByHIVstage[ia][1] * 
					a->PropnByStage1[ia][iz];
				a->NumberByStage2[ia][iz] = TempNumbersByHIVstage[ia][2] * 
					a->PropnByStage2[ia][iz];
				a->NumberByStage3[ia][iz] = TempNumbersByHIVstage[ia][3] * 
					a->PropnByStage3[ia][iz];
				a->NumberByStage4[ia][iz] = TempNumbersByHIVstage[ia][4] * 
					a->PropnByStage4[ia][iz];
				a->NumberByStage5[ia][iz] = TempNumbersByHIVstage[ia][5] * 
					a->PropnByStage5[ia][iz];
			}
		}
	}
}

void SexuallyExpM::GetNumbersRemaining()
{
	int ia, is;

	for(ia=0; ia<16; ia++){
		TempNumbersByHIVstage[ia][0] = NumbersByHIVstage[ia][0] * Remain[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				TempNumbersByHIVstage[ia][is] = NumbersByHIVstage[ia][is] * Remain[ia][is];}
		}
	}
	if(HSVind==1){GetNumbersRemaining1(&MHSV);}
	if(TPind==1){GetNumbersRemaining1(&MTP);}
	if(HDind==1){GetNumbersRemaining1(&MHD);}
	if(NGind==1){GetNumbersRemaining1(&MNG);}
	if(CTind==1){GetNumbersRemaining1(&MCT);}
	if(TVind==1){GetNumbersRemaining1(&MTV);}
}

void SexuallyExpM::CalcAllAgeChanges()
{
	int ia, is;

	for(ia=0; ia<15; ia++){
		if(VirginInd==0){
			NumbersByHIVstage[15-ia][0] = NumbersByHIVstage[15-ia][0] * (1.0 - 
				AgeExitRateM[15-ia][0]) + NumbersByHIVstage[14-ia][0] * AgeExitRateM[14-ia][0];}
		else{
			NumbersByHIVstage[15-ia][0] = NumbersByHIVstage[15-ia][0] * (1.0 - 
				VirginAgeExitRate[15-ia][0]) + NumbersByHIVstage[14-ia][0] * 
				VirginAgeExitRate[14-ia][0];}
		if(HIVind==1 && VirginInd==0){
			for(is=1; is<6; is++){
				NumbersByHIVstage[15-ia][is] = NumbersByHIVstage[15-ia][is] * (1.0 - 
					AgeExitRateM[15-ia][is]) + NumbersByHIVstage[14-ia][is] * 
					AgeExitRateM[14-ia][is];}
		}
	}
	// Repeat the above code for the first age category (no age entrants).
	if(VirginInd==0){
		NumbersByHIVstage[0][0] = NumbersByHIVstage[0][0] * (1.0 - AgeExitRateM[0][0]);}
	else{
		NumbersByHIVstage[0][0] = NumbersByHIVstage[0][0] * (1.0 - VirginAgeExitRate[0][0]);}
	if(HIVind==1 && VirginInd==0){
		for(is=1; is<6; is++){
			NumbersByHIVstage[0][is] = NumbersByHIVstage[0][is] * (1.0 - AgeExitRateM[0][is]);}
	}

	// Calculate TotalAlive
	for(ia=0; ia<16; ia++){
		TotalAlive[ia] = NumbersByHIVstage[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				TotalAlive[ia] += NumbersByHIVstage[ia][is];}
		}
	}

	if(HSVind==1){CalcAgeChanges(&MHSV);}
	if(TPind==1){CalcAgeChanges(&MTP);}
	if(HDind==1){CalcAgeChanges(&MHD);}
	if(NGind==1){CalcAgeChanges(&MNG);}
	if(CTind==1){CalcAgeChanges(&MCT);}
	if(TVind==1){CalcAgeChanges(&MTV);}
}

void SexuallyExpM::GetTotalBySTDstage()
{
	if(HSVind==1){MHSV.GetTotalBySTDstage(&HSVtransitionM);}
	if(TPind==1){MTP.GetTotalBySTDstage(&TPtransitionM);}
	if(HDind==1){MHD.GetTotalBySTDstage(&HDtransitionM);}
	if(NGind==1){MNG.GetTotalBySTDstage(&NGtransitionM);}
	if(CTind==1){MCT.GetTotalBySTDstage(&CTtransitionM);}
	if(TVind==1){MTV.GetTotalBySTDstage(&TVtransitionM);}
}

void SexuallyExpM::RecordPropnsByStage(ofstream* file)
{
	int ia, is, lim;

	if(VirginInd==1){
		lim = 4;}
	else{lim = 16;}

	for(ia=0; ia<lim; ia++){
		for(is=0; is<MNG.nStates-1; is++){
			*file<<setw(10)<<right<<MNG.PropnByStage0[ia][is]<<"	";}
		*file<<setw(10)<<right<<MNG.PropnByStage0[ia][MNG.nStates-1]<<endl;
	}
	*file<<endl;
}

SexuallyExpF::SexuallyExpF()
{
	SexInd = 1;
	FSWind = 0;
}

void SexuallyExpF::Reset()
{
	int ia, is, iz;

	for(ia=0; ia<16; ia++){
		NumbersByHIVstage[ia][0] = TotalAlive[ia];
		if(HIVind==1){
			for(is=1; is<6; is++){
				NumbersByHIVstage[ia][is] = 0;}
		}
	}

	if(HIVind==1 && RiskGroup==1){
		for(ia=1; ia<8; ia++){
			NumbersByHIVstage[ia][0] = TotalAlive[ia] * (1.0 - InitHIVprevHigh);
			NumbersByHIVstage[ia][2] = TotalAlive[ia] * InitHIVprevHigh;
			for(iz=0; iz<7; iz++){
				FHSV.PropnByStage2[ia][iz] = FHSV.PropnByStage0[ia][iz];
				FTP.PropnByStage2[ia][iz] = FTP.PropnByStage0[ia][iz];
				FHD.PropnByStage2[ia][iz] = FHD.PropnByStage0[ia][iz];
				FNG.PropnByStage2[ia][iz] = FNG.PropnByStage0[ia][iz];
				FCT.PropnByStage2[ia][iz] = FCT.PropnByStage0[ia][iz];
				FTV.PropnByStage2[ia][iz] = FTV.PropnByStage0[ia][iz];
				FBV.PropnByStage2[ia][iz] = FBV.PropnByStage0[ia][iz];
				FVC.PropnByStage2[ia][iz] = FVC.PropnByStage0[ia][iz];
			}
		}
	}
}

void SexuallyExpF::GetAllNumbersBySTDstage()
{
	if(TPind==1){GetNumbersBySTDstage(&FTP);}
	if(HSVind==1){GetNumbersBySTDstage(&FHSV);}
	if(HDind==1){GetNumbersBySTDstage(&FHD);}
	if(NGind==1){GetNumbersBySTDstage(&FNG);}
	if(CTind==1){GetNumbersBySTDstage(&FCT);}
	if(TVind==1){GetNumbersBySTDstage(&FTV);}
	if(BVind==1){GetNumbersBySTDstage(&FBV);}
	if(VCind==1){GetNumbersBySTDstage(&FVC);}
}

void SexuallyExpF::GetAllPropnsBySTDstage()
{
	if(TPind==1){GetPropnsBySTDstage(&FTP);}
	if(HSVind==1){GetPropnsBySTDstage(&FHSV);}
	if(HDind==1){GetPropnsBySTDstage(&FHD);}
	if(NGind==1){GetPropnsBySTDstage(&FNG);}
	if(CTind==1){GetPropnsBySTDstage(&FCT);}
	if(TVind==1){GetPropnsBySTDstage(&FTV);}
	if(BVind==1){GetPropnsBySTDstage(&FBV);}
	if(VCind==1){GetPropnsBySTDstage(&FVC);}
}

void SexuallyExpF::UpdateSyndromePropns()
{
	int ia;

	for(ia=0; ia<16; ia++){
		GUDpropn[ia][0] = 1.0 - (1.0 - FHSV.PropnByStage0[ia][1] - FHSV.PropnByStage0[ia][3])
			* (1 - FTP.PropnByStage0[ia][2]) * (1.0 - FHD.PropnByStage0[ia][1]);
		GUDpropn[ia][1] = 1.0 - (1.0 - FHSV.PropnByStage1[ia][1] - FHSV.PropnByStage1[ia][3])
			* (1 - FTP.PropnByStage1[ia][2]) * (1.0 - FHD.PropnByStage1[ia][1]);
		GUDpropn[ia][2] = 1.0 - (1.0 - FHSV.PropnByStage2[ia][1] - FHSV.PropnByStage2[ia][3])
			* (1 - FTP.PropnByStage2[ia][2]) * (1.0 - FHD.PropnByStage2[ia][1]);
		GUDpropn[ia][3] = 1.0 - (1.0 - FHSV.PropnByStage3[ia][1] - FHSV.PropnByStage3[ia][3])
			* (1 - FTP.PropnByStage3[ia][2]) * (1.0 - FHD.PropnByStage3[ia][1]);
		GUDpropn[ia][4] = 1.0 - (1.0 - FHSV.PropnByStage4[ia][1] - FHSV.PropnByStage4[ia][3])
			* (1 - FTP.PropnByStage4[ia][2]) * (1.0 - FHD.PropnByStage4[ia][1]);
		GUDpropn[ia][5] = 1.0 - (1.0 - FHSV.PropnByStage5[ia][1] - FHSV.PropnByStage5[ia][3])
			* (1 - FTP.PropnByStage5[ia][2]) * (1.0 - FHD.PropnByStage5[ia][1]);
		DischargePropn[ia][0] = (1 - GUDpropn[ia][0])*(1.0 - (1.0 - FNG.PropnByStage0[ia][1])
			* (1.0 - FCT.PropnByStage0[ia][1]) * (1.0 - FTV.PropnByStage0[ia][1]) * 
			(1.0 - FVC.PropnByStage0[ia][2]) * (1.0 - FBV.PropnByStage0[ia][2]));
		DischargePropn[ia][1] = (1 - GUDpropn[ia][1])*(1.0 - (1.0 - FNG.PropnByStage1[ia][1])
			* (1.0 - FCT.PropnByStage1[ia][1]) * (1.0 - FTV.PropnByStage1[ia][1]) * 
			(1.0 - FVC.PropnByStage1[ia][2]) * (1.0 - FBV.PropnByStage1[ia][2]));
		DischargePropn[ia][2] = (1 - GUDpropn[ia][2])*(1.0 - (1.0 - FNG.PropnByStage2[ia][1])
			* (1.0 - FCT.PropnByStage2[ia][1]) * (1.0 - FTV.PropnByStage2[ia][1]) * 
			(1.0 - FVC.PropnByStage2[ia][2]) * (1.0 - FBV.PropnByStage2[ia][2]));
		DischargePropn[ia][3] = (1 - GUDpropn[ia][3])*(1.0 - (1.0 - FNG.PropnByStage3[ia][1])
			* (1.0 - FCT.PropnByStage3[ia][1]) * (1.0 - FTV.PropnByStage3[ia][1]) * 
			(1.0 - FVC.PropnByStage3[ia][2]) * (1.0 - FBV.PropnByStage3[ia][2]));
		DischargePropn[ia][4] = (1 - GUDpropn[ia][4])*(1.0 - (1.0 - FNG.PropnByStage4[ia][1])
			* (1.0 - FCT.PropnByStage4[ia][1]) * (1.0 - FTV.PropnByStage4[ia][1]) * 
			(1.0 - FVC.PropnByStage4[ia][2]) * (1.0 - FBV.PropnByStage4[ia][2]));
		DischargePropn[ia][5] = (1 - GUDpropn[ia][5])*(1.0 - (1.0 - FNG.PropnByStage5[ia][1])
			* (1.0 - FCT.PropnByStage5[ia][1]) * (1.0 - FTV.PropnByStage5[ia][1]) * 
			(1.0 - FVC.PropnByStage5[ia][2]) * (1.0 - FBV.PropnByStage5[ia][2]));
		AsympSTDpropn[ia][0] = (1.0 - (FHSV.PropnByStage0[ia][0] + FHSV.PropnByStage0[ia][2] 
			+ FHSV.PropnByStage0[ia][4])*(FTP.PropnByStage0[ia][0] + FTP.PropnByStage0[ia][4]
			+ FTP.PropnByStage0[ia][5] + FTP.PropnByStage0[ia][6]) * FHD.PropnByStage0[ia][0]
			* FNG.PropnByStage0[ia][0] * FCT.PropnByStage0[ia][0] * FTV.PropnByStage0[ia][0]
			* FVC.PropnByStage0[ia][0]*(FBV.PropnByStage0[ia][0] + FBV.PropnByStage0[ia][1]))
			- GUDpropn[ia][0] - DischargePropn[ia][0];
		AsympSTDpropn[ia][1] = (1.0 - (FHSV.PropnByStage1[ia][0] + FHSV.PropnByStage1[ia][2] 
			+ FHSV.PropnByStage1[ia][4])*(FTP.PropnByStage1[ia][0] + FTP.PropnByStage1[ia][4]
			+ FTP.PropnByStage1[ia][5] + FTP.PropnByStage1[ia][6]) * FHD.PropnByStage1[ia][0]
			* FNG.PropnByStage1[ia][0] * FCT.PropnByStage1[ia][0] * FTV.PropnByStage1[ia][0]
			* FVC.PropnByStage1[ia][0]*(FBV.PropnByStage1[ia][0] + FBV.PropnByStage1[ia][1]))
			- GUDpropn[ia][1] - DischargePropn[ia][1];
		AsympSTDpropn[ia][2] = (1.0 - (FHSV.PropnByStage2[ia][0] + FHSV.PropnByStage2[ia][2] 
			+ FHSV.PropnByStage2[ia][4])*(FTP.PropnByStage2[ia][0] + FTP.PropnByStage2[ia][4]
			+ FTP.PropnByStage2[ia][5] + FTP.PropnByStage2[ia][6]) * FHD.PropnByStage2[ia][0]
			* FNG.PropnByStage2[ia][0] * FCT.PropnByStage2[ia][0] * FTV.PropnByStage2[ia][0]
			* FVC.PropnByStage2[ia][0]*(FBV.PropnByStage2[ia][0] + FBV.PropnByStage2[ia][1]))
			- GUDpropn[ia][2] - DischargePropn[ia][2];
		AsympSTDpropn[ia][3] = (1.0 - (FHSV.PropnByStage3[ia][0] + FHSV.PropnByStage3[ia][2] 
			+ FHSV.PropnByStage3[ia][4])*(FTP.PropnByStage3[ia][0] + FTP.PropnByStage3[ia][4]
			+ FTP.PropnByStage3[ia][5] + FTP.PropnByStage3[ia][6]) * FHD.PropnByStage3[ia][0]
			* FNG.PropnByStage3[ia][0] * FCT.PropnByStage3[ia][0] * FTV.PropnByStage3[ia][0]
			* FVC.PropnByStage3[ia][0]*(FBV.PropnByStage3[ia][0] + FBV.PropnByStage3[ia][1]))
			- GUDpropn[ia][3] - DischargePropn[ia][3];
		AsympSTDpropn[ia][4] = (1.0 - (FHSV.PropnByStage4[ia][0] + FHSV.PropnByStage4[ia][2] 
			+ FHSV.PropnByStage4[ia][4])*(FTP.PropnByStage4[ia][0] + FTP.PropnByStage4[ia][4]
			+ FTP.PropnByStage4[ia][5] + FTP.PropnByStage4[ia][6]) * FHD.PropnByStage4[ia][0]
			* FNG.PropnByStage4[ia][0] * FCT.PropnByStage4[ia][0] * FTV.PropnByStage4[ia][0]
			* FVC.PropnByStage4[ia][0]*(FBV.PropnByStage4[ia][0] + FBV.PropnByStage4[ia][1]))
			- GUDpropn[ia][4] - DischargePropn[ia][4];
		AsympSTDpropn[ia][5] = (1.0 - (FHSV.PropnByStage5[ia][0] + FHSV.PropnByStage5[ia][2] 
			+ FHSV.PropnByStage5[ia][4])*(FTP.PropnByStage5[ia][0] + FTP.PropnByStage5[ia][4]
			+ FTP.PropnByStage5[ia][5] + FTP.PropnByStage5[ia][6]) * FHD.PropnByStage5[ia][0]
			* FNG.PropnByStage5[ia][0] * FCT.PropnByStage5[ia][0] * FTV.PropnByStage5[ia][0]
			* FVC.PropnByStage5[ia][0]*(FBV.PropnByStage5[ia][0] + FBV.PropnByStage5[ia][1]))
			- GUDpropn[ia][5] - DischargePropn[ia][5];
	}
}

void SexuallyExpF::GetAllSTDcofactors()
{
	int ia, is;

	for(ia=0; ia<16; ia++){
		for(is=0; is<6; is++){
			STDcofactor[ia][is] = 1.0;} // Default (no STD cofactors)
	}

	if(CofactorType==1){ // Multiplicative cofactors
		if(TPind==1){GetSTDcofactor(&FTP, &TPtransitionF);}
		if(HSVind==1){GetSTDcofactor(&FHSV, &HSVtransitionF);}
		if(HDind==1){GetSTDcofactor(&FHD, &HDtransitionF);}
		if(NGind==1){GetSTDcofactor(&FNG, &NGtransitionF);}
		if(CTind==1){GetSTDcofactor(&FCT, &CTtransitionF);}
		if(TVind==1){GetSTDcofactor(&FTV, &TVtransitionF);}
		if(BVind==1){GetSTDcofactor(&FBV, &BVtransitionF);}
		if(VCind==1){GetSTDcofactor(&FVC, &VCtransitionF);}
	}

	else if(CofactorType==2){ // Saturation cofactors
		for(ia=0; ia<16; ia++){
			STDcofactor[ia][0] += GUDpropn[ia][0] * SuscepIncreaseSyndrome[0][1] +
				DischargePropn[ia][0] * SuscepIncreaseSyndrome[1][1] +
				AsympSTDpropn[ia][0] * SuscepIncreaseSyndrome[2][1];
			for(is=1; is<6; is++){
				STDcofactor[ia][is] += GUDpropn[ia][is] * InfecIncreaseSyndrome[0][1] +
					DischargePropn[ia][is] * InfecIncreaseSyndrome[1][1] +
					AsympSTDpropn[ia][is] * InfecIncreaseSyndrome[2][1];}
		}
	}
}

void SexuallyExpF::CalcTransmissionProb()
{
	int ia, is;
	double sumx;

	if(TPind==1){FTP.CalcTransmissionProb(&TPtransitionF);}
	if(HSVind==1){FHSV.CalcTransmissionProb(&HSVtransitionF);}
	if(HDind==1){FHD.CalcTransmissionProb(&HDtransitionF);}
	if(NGind==1){FNG.CalcTransmissionProb(&NGtransitionF);}
	if(CTind==1){FCT.CalcTransmissionProb(&CTtransitionF);}
	if(TVind==1){FTV.CalcTransmissionProb(&TVtransitionF);}

	if(HIVind==1){
		for(ia=0; ia<16; ia++){
			sumx = 0;
			for(is=1; is<6; is++){
				sumx += NumbersByHIVstage[ia][is] * (1.0 + 
					HIVtransitionF.HIVinfecIncrease[is-1]) * STDcofactor[ia][is];}
			// FSW-client transmission prob
			if(FSWind==1){
				WeightedProbTransm[ia][0] = sumx * HIVtransitionF.TransmProb[0] *
					RatioAsympToAveF;}
			// ST relationship transmission prob
			if(NumberPartnersS1>0 && RiskGroup==1){
				WeightedProbTransm[ia][1] = sumx * HIVtransitionF.TransmProb[1] *
					RatioAsympToAveF;}
			if(NumberPartnersS2>0 && RiskGroup==2){
				WeightedProbTransm[ia][3] = sumx * HIVtransitionF.TransmProb[3] *
					RatioAsympToAveF;}
			if((NumberPartnersS1>0 && RiskGroup==2) || (NumberPartnersS2>0 && RiskGroup==1)){
				WeightedProbTransm[ia][2] = sumx * HIVtransitionF.TransmProb[2] *
					RatioAsympToAveF;}
			// LT relationship transmission prob
			if(NumberPartnersL1>0 && RiskGroup==1){
				WeightedProbTransm[ia][4] = sumx * HIVtransitionF.TransmProb[4] *
					RatioAsympToAveF;}
			if(NumberPartnersL2>0 && RiskGroup==2){
				WeightedProbTransm[ia][6] = sumx * HIVtransitionF.TransmProb[6] *
					RatioAsympToAveF;}
			if((NumberPartnersL1>0 && RiskGroup==2) || (NumberPartnersL2>0 && RiskGroup==1)){
				WeightedProbTransm[ia][5] = sumx * HIVtransitionF.TransmProb[5] *
					RatioAsympToAveF;}
		}
	}

	if(TPind==1){CalcOneTransmProb(&TPtransitionF, &FTP);}
	if(HSVind==1){CalcOneTransmProb(&HSVtransitionF, &FHSV);}
	if(HDind==1){CalcOneTransmProb(&HDtransitionF, &FHD);}
	if(NGind==1){CalcOneTransmProb(&NGtransitionF, &FNG);}
	if(CTind==1){CalcOneTransmProb(&CTtransitionF, &FCT);}
	if(TVind==1){CalcOneTransmProb(&TVtransitionF, &FTV);}
	if(HIVind==1){CalcHIVtransmProb(&HIVtransitionF);}
}

void SexuallyExpF::CalcOneTransmProb(STDtransition* a, TradnalSTD* b)
{
	int ia;

	for(ia=0; ia<16; ia++){
		if(RiskGroup==1){
			a->TransmProbS1to1[ia] += b->WeightedProbTransm[ia] * NumberPartnersS1;
			a->TransmProbS1to2[ia] += b->WeightedProbTransm[ia] * NumberPartnersS2;
			a->TransmProbL1to1[ia] += b->WeightedProbTransm[ia] * NumberPartnersL1;
			a->TransmProbL1to2[ia] += b->WeightedProbTransm[ia] * NumberPartnersL2;
			if(FSWind==1){
				a->InfectProbFSW[0] += b->WeightedProbTransm[ia];
			}
		}
		else{
			a->TransmProbS2to1[ia] += b->WeightedProbTransm[ia] * NumberPartnersS1;
			a->TransmProbS2to2[ia] += b->WeightedProbTransm[ia] * NumberPartnersS2;
			a->TransmProbL2to1[ia] += b->WeightedProbTransm[ia] * NumberPartnersL1;
			a->TransmProbL2to2[ia] += b->WeightedProbTransm[ia] * NumberPartnersL2;
		}
	}
}

void SexuallyExpF::CalcInfectionProb()
{
	if(HIVind==1){CalcHIVinfectProb(&HIVtransitionM);}
	if(HSVind==1){CalcHSVinfectProb(&HSVtransitionM, &FHSV);}
	if(TPind==1){CalcOneInfectProb(&TPtransitionM, &FTP);}
	if(HDind==1){CalcOneInfectProb(&HDtransitionM, &FHD);}
	if(NGind==1){CalcOneInfectProb(&NGtransitionM, &FNG);}
	if(CTind==1){CalcOneInfectProb(&CTtransitionM, &FCT);}
	if(TVind==1){CalcOneInfectProb(&TVtransitionM, &FTV);}
}

void SexuallyExpF::CalcOneInfectProb(STDtransition* a, TradnalSTD* b)
{
	int ia;

	if(RiskGroup==1){
		for(ia=0; ia<16; ia++){
			b->InfectProb[ia] = 1.0;
			if(NumberPartnersS1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS1from1[ia], NumberPartnersS1 * 
					FreqSexST[ia][1] * CycleS/CycleD);}
			if(NumberPartnersS2>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS1from2[ia], NumberPartnersS2 * 
					FreqSexST[ia][1] * CycleS/CycleD);}
			if(NumberPartnersL1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL1from1[ia], 
					FreqSexLT[ia][1] * CycleS/CycleD);}
			if(NumberPartnersL2>0 && NoBacterialTransm12==0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL1from2[ia], 
					FreqSexLT[ia][1] * CycleS/CycleD);}
			if(FSWind==1){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbFSW[ia], 
					AnnNumberClients/CycleD);}
			b->InfectProb[ia] = 1.0 - b->InfectProb[ia];
		}
	}
	else{
		for(ia=0; ia<16; ia++){
			b->InfectProb[ia] = 1.0;
			if(NumberPartnersS1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS2from1[ia], 
					FreqSexST[ia][1] * CycleS/CycleD);}
			if(NumberPartnersS2>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS2from2[ia], 
					FreqSexST[ia][1] * CycleS/CycleD);}
			if(NumberPartnersL1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL2from1[ia], 
					FreqSexLT[ia][1] * CycleS/CycleD);}
			if(NumberPartnersL2>0 && NoBacterialTransm22==0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL2from2[ia],  
					FreqSexLT[ia][1] * CycleS/CycleD);}
			b->InfectProb[ia] = 1.0 - b->InfectProb[ia];
		}
	}
}

void SexuallyExpF::CalcHSVinfectProb(HerpesTransition* a, Herpes* b)
{
	// The same as the CalcOneInfectProb function, except that the arguments are different 
	// and we use NoViralTransm in place of NoBacterialTransm.

	int ia;

	if(RiskGroup==1){
		for(ia=0; ia<16; ia++){
			b->InfectProb[ia] = 1.0;
			if(NumberPartnersS1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS1from1[ia], NumberPartnersS1 * 
					FreqSexST[ia][1] * CycleS/CycleD);}
			if(NumberPartnersS2>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS1from2[ia], NumberPartnersS2 * 
					FreqSexST[ia][1] * CycleS/CycleD);}
			if(NumberPartnersL1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL1from1[ia], 
					FreqSexLT[ia][1] * CycleS/CycleD);}
			if(NumberPartnersL2>0 && NoViralTransm12==0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL1from2[ia], 
					FreqSexLT[ia][1] * CycleS/CycleD);}
			if(FSWind==1){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbFSW[ia], 
					AnnNumberClients/CycleD);}
			b->InfectProb[ia] = 1.0 - b->InfectProb[ia];
		}
	}
	else{
		for(ia=0; ia<16; ia++){
			b->InfectProb[ia] = 1.0;
			if(NumberPartnersS1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS2from1[ia], 
					FreqSexST[ia][1] * CycleS/CycleD);}
			if(NumberPartnersS2>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbS2from2[ia], 
					FreqSexST[ia][1] * CycleS/CycleD);}
			if(NumberPartnersL1>0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL2from1[ia], 
					FreqSexLT[ia][1] * CycleS/CycleD);}
			if(NumberPartnersL2>0 && NoViralTransm22==0){
				b->InfectProb[ia] *= pow(1.0 - a->InfectProbL2from2[ia],  
					FreqSexLT[ia][1] * CycleS/CycleD);}
			b->InfectProb[ia] = 1.0 - b->InfectProb[ia];
		}
	}
}

void SexuallyExpF::CalcHIVinfectProb(HIVtransition* a)
{
	// Similar to the CalcHSVinfectProb function, except that the arguments are different, 
	// we allow for STD cofactors, we use HIVinfectProb instead of b->InfectProb, and we
	// have to impose a max on the transmission prob per sex act (because of STD cofactors).

	int ia;
	double ProbPerAct;

	if(RiskGroup==1){
		for(ia=0; ia<16; ia++){
			HIVinfectProb[ia] = 1.0;
			//HIVinfectProbST[ia] = 1.0;
			//HIVinfectProbLT[ia] = 1.0;
			//HIVinfectProbCS[ia] = 1.0;
			if(NumberPartnersS1>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbS1from1[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, NumberPartnersS1 *  
					FreqSexST[ia][1] * CycleS/CycleD);
				//HIVinfectProbST[ia] *= pow(1.0 - ProbPerAct, NumberPartnersS1 *  
				//	FreqSexST[ia][1] * CycleS/CycleD);
			}
			if(NumberPartnersS2>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbS1from2[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, NumberPartnersS2 * 
					FreqSexST[ia][1] * CycleS/CycleD);
				//HIVinfectProbST[ia] *= pow(1.0 - ProbPerAct, NumberPartnersS2 * 
				//	FreqSexST[ia][1] * CycleS/CycleD);
			}
			if(NumberPartnersL1>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbL1from1[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][1] * CycleS/CycleD);
				//HIVinfectProbLT[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][1] * CycleS/CycleD);
			}
			if(NumberPartnersL2>0 && NoViralTransm12==0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbL1from2[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][1] * CycleS/CycleD);
				//HIVinfectProbLT[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][1] * CycleS/CycleD);
			}
			if(FSWind==1){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbFSW[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, AnnNumberClients/CycleD);
				//HIVinfectProbCS[ia] *= pow(1.0 - ProbPerAct, AnnNumberClients/CycleD);
			}
			HIVinfectProb[ia] = 1.0 - HIVinfectProb[ia];
			//HIVinfectProbST[ia] = 1.0 - HIVinfectProbST[ia];
			//HIVinfectProbLT[ia] = 1.0 - HIVinfectProbLT[ia];
			//HIVinfectProbCS[ia] = 1.0 - HIVinfectProbCS[ia];
		}
	}
	else{
		for(ia=0; ia<16; ia++){
			HIVinfectProb[ia] = 1.0;
			//HIVinfectProbST[ia] = 1.0;
			//HIVinfectProbLT[ia] = 1.0;
			if(NumberPartnersS1>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbS2from1[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexST[ia][1] * CycleS/CycleD);
				//HIVinfectProbST[ia] *= pow(1.0 - ProbPerAct, FreqSexST[ia][1] * CycleS/CycleD);
			}
			if(NumberPartnersS2>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbS2from2[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexST[ia][1] * CycleS/CycleD);
				//HIVinfectProbST[ia] *= pow(1.0 - ProbPerAct, FreqSexST[ia][1] * CycleS/CycleD);
			}
			if(NumberPartnersL1>0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbL2from1[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][1] * CycleS/CycleD);
				//HIVinfectProbLT[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][1] * CycleS/CycleD);
			}
			if(NumberPartnersL2>0 && NoViralTransm22==0){
				ProbPerAct = STDcofactor[ia][0] * a->InfectProbL2from2[ia];
				if(ProbPerAct>MaxHIVprev){ProbPerAct = MaxHIVprev;}
				HIVinfectProb[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][1] * CycleS/CycleD);
				//HIVinfectProbLT[ia] *= pow(1.0 - ProbPerAct, FreqSexLT[ia][1] * CycleS/CycleD);
			}
			HIVinfectProb[ia] = 1.0 - HIVinfectProb[ia];
			//HIVinfectProbST[ia] = 1.0 - HIVinfectProbST[ia];
			//HIVinfectProbLT[ia] = 1.0 - HIVinfectProbLT[ia];
		}
	}
}

void SexuallyExpF::CalcSTDtransitions()
{
	if(HSVind==1){FHSV.CalcSTDtransitions(&HSVtransitionF, FSWind);}
	if(TPind==1){FTP.CalcSTDtransitions(&TPtransitionF, FSWind);}
	if(HDind==1){FHD.CalcSTDtransitions(&HDtransitionF, FSWind);}
	if(NGind==1){FNG.CalcSTDtransitions(&NGtransitionF, FSWind);}
	if(CTind==1){FCT.CalcSTDtransitions(&CTtransitionF, FSWind);}
	if(TVind==1){FTV.CalcSTDtransitions(&TVtransitionF, FSWind);}
	if(VCind==1){FVC.CalcSTDtransitions(&VCtransitionF, FSWind);}
	if(BVind==1){FBV.CalcSTDtransitions(&BVtransitionF, FSWind, NumberPartners);}
	if(HIVind==1){CalcHIVtransitions(&HIVtransitionF);}
}

void SexuallyExpF::AllHIVstageChanges()
{
	if(HSVind==1){HIVstageChanges(&FHSV);}
	if(TPind==1){HIVstageChanges(&FTP);}
	if(HDind==1){HIVstageChanges(&FHD);}
	if(NGind==1){HIVstageChanges(&FNG);}
	if(CTind==1){HIVstageChanges(&FCT);}
	if(TVind==1){HIVstageChanges(&FTV);}
	if(VCind==1){HIVstageChanges(&FVC);}
	if(BVind==1){HIVstageChanges(&FBV);}
}

void SexuallyExpF::GetNumbersRemaining1(NonHIV* a)
{
	int ia, iz;

	for(ia=0; ia<16; ia++){
		for(iz=0; iz<a->nStates; iz++){
			a->NumberByStage0[ia][iz] = TempNumbersByHIVstage[ia][0] * 
				a->PropnByStage0[ia][iz];
			if(HIVind==1){
				a->NumberByStage1[ia][iz] = TempNumbersByHIVstage[ia][1] * 
					a->PropnByStage1[ia][iz];
				a->NumberByStage2[ia][iz] = TempNumbersByHIVstage[ia][2] * 
					a->PropnByStage2[ia][iz];
				a->NumberByStage3[ia][iz] = TempNumbersByHIVstage[ia][3] * 
					a->PropnByStage3[ia][iz];
				a->NumberByStage4[ia][iz] = TempNumbersByHIVstage[ia][4] * 
					a->PropnByStage4[ia][iz];
				a->NumberByStage5[ia][iz] = TempNumbersByHIVstage[ia][5] * 
					a->PropnByStage5[ia][iz];
			}
		}
	}
}

void SexuallyExpF::GetNumbersRemaining()
{
	int ia, is;

	for(ia=0; ia<16; ia++){
		TempNumbersByHIVstage[ia][0] = NumbersByHIVstage[ia][0] * Remain[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				TempNumbersByHIVstage[ia][is] = NumbersByHIVstage[ia][is] * Remain[ia][is];}
		}
	}
	if(HSVind==1){GetNumbersRemaining1(&FHSV);}
	if(TPind==1){GetNumbersRemaining1(&FTP);}
	if(HDind==1){GetNumbersRemaining1(&FHD);}
	if(NGind==1){GetNumbersRemaining1(&FNG);}
	if(CTind==1){GetNumbersRemaining1(&FCT);}
	if(TVind==1){GetNumbersRemaining1(&FTV);}
	if(VCind==1){GetNumbersRemaining1(&FVC);}
	if(BVind==1){GetNumbersRemaining1(&FBV);}
}

void SexuallyExpF::CalcAllAgeChanges()
{
	int ia, is;

	for(ia=0; ia<15; ia++){
		if(VirginInd==0){
			NumbersByHIVstage[15-ia][0] = NumbersByHIVstage[15-ia][0] * (1.0 - 
				AgeExitRateF[15-ia][0]) + NumbersByHIVstage[14-ia][0] * AgeExitRateF[14-ia][0];}
		else{
			NumbersByHIVstage[15-ia][0] = NumbersByHIVstage[15-ia][0] * (1.0 - 
				VirginAgeExitRate[15-ia][1]) + NumbersByHIVstage[14-ia][0] * 
				VirginAgeExitRate[14-ia][1];}
		if(HIVind==1 && VirginInd==0){
			for(is=1; is<6; is++){
				NumbersByHIVstage[15-ia][is] = NumbersByHIVstage[15-ia][is] * (1.0 - 
					AgeExitRateF[15-ia][is]) + NumbersByHIVstage[14-ia][is] * 
					AgeExitRateF[14-ia][is];}
		}
	}
	// Repeat the above code for the first age category (no age entrants).
	if(VirginInd==0){
		NumbersByHIVstage[0][0] = NumbersByHIVstage[0][0] * (1.0 - AgeExitRateF[0][0]);}
	else{
		NumbersByHIVstage[0][0] = NumbersByHIVstage[0][0] * (1.0 - VirginAgeExitRate[0][1]);}
	if(HIVind==1 && VirginInd==0){
		for(is=1; is<6; is++){
			NumbersByHIVstage[0][is] = NumbersByHIVstage[0][is] * (1.0 - AgeExitRateF[0][is]);}
	}

	// Calculate TotalAlive
	for(ia=0; ia<16; ia++){
		TotalAlive[ia] = NumbersByHIVstage[ia][0];
		if(HIVind==1){
			for(is=1; is<6; is++){
				TotalAlive[ia] += NumbersByHIVstage[ia][is];}
		}
	}

	if(HSVind==1){CalcAgeChanges(&FHSV);}
	if(TPind==1){CalcAgeChanges(&FTP);}
	if(HDind==1){CalcAgeChanges(&FHD);}
	if(NGind==1){CalcAgeChanges(&FNG);}
	if(CTind==1){CalcAgeChanges(&FCT);}
	if(TVind==1){CalcAgeChanges(&FTV);}
	if(VCind==1){CalcAgeChanges(&FVC);}
	if(BVind==1){CalcAgeChanges(&FBV);}
}

void SexuallyExpF::GetTotalBySTDstage()
{
	if(HSVind==1){FHSV.GetTotalBySTDstage(&HSVtransitionF);}
	if(TPind==1){FTP.GetTotalBySTDstage(&TPtransitionF);}
	if(HDind==1){FHD.GetTotalBySTDstage(&HDtransitionF);}
	if(NGind==1){FNG.GetTotalBySTDstage(&NGtransitionF);}
	if(CTind==1){FCT.GetTotalBySTDstage(&CTtransitionF);}
	if(TVind==1){FTV.GetTotalBySTDstage(&TVtransitionF);}
	if(VCind==1){FVC.GetTotalBySTDstage(&VCtransitionF);}
	if(BVind==1){FBV.GetTotalBySTDstage(&BVtransitionF);}
}

void SexuallyExpF::GetCSWprev(STDtransition* a, NonHIV* b, int BVindicator)
{
	int ia;
	double numerator, denominator;

	numerator = 0;
	denominator = 0;
	for(ia=0; ia<16; ia++){
		numerator += b->NumberByStage0[ia][0];
		if(BVindicator==1){numerator += b->NumberByStage0[ia][1];}
		if(HIVind==1){
			numerator += b->NumberByStage1[ia][0] + b->NumberByStage2[ia][0] +
				b->NumberByStage3[ia][0] + b->NumberByStage4[ia][0] + b->NumberByStage5[ia][0];
			if(BVindicator==1){
				numerator += b->NumberByStage1[ia][1] + b->NumberByStage2[ia][1] +
					b->NumberByStage3[ia][1] + b->NumberByStage4[ia][1] + 
					b->NumberByStage5[ia][1];}
		}
		denominator += TotalAlive[ia];
	}
	a->CSWprevalence = 1.0 - numerator/denominator;
}

void SexuallyExpF::RecordPropnsByStage(ofstream* file)
{
	int ia, is, lim;

	if(VirginInd==1){
		lim = 4;}
	else{lim = 16;}

	for(ia=0; ia<lim; ia++){
		for(is=0; is<FNG.nStates-1; is++){
			*file<<setw(10)<<right<<FNG.PropnByStage0[ia][is]<<"	";}
		*file<<setw(10)<<right<<FNG.PropnByStage0[ia][FNG.nStates-1]<<endl;
	}
	*file<<endl;
}

SexCohort::SexCohort(){}

SexCohortM::SexCohortM(int Risk)
{
	SexInd = 0;
	RiskGroup = Risk;
	
	Virgin.RiskGroup = Risk;
	NoPartner.RiskGroup = Risk;
	S1.RiskGroup = Risk;
	S2.RiskGroup = Risk;
	L1.RiskGroup = Risk;
	L2.RiskGroup = Risk;
	S11.RiskGroup = Risk;
	S12.RiskGroup = Risk;
	S22.RiskGroup = Risk;
	L11.RiskGroup = Risk;
	L12.RiskGroup = Risk;
	L21.RiskGroup = Risk;
	L22.RiskGroup = Risk;

	Virgin.SetPartnerNumbers(0, 0, 0, 0);
	NoPartner.SetPartnerNumbers(0, 0, 0, 0);
	S1.SetPartnerNumbers(1, 0, 0, 0);
	S2.SetPartnerNumbers(0, 1, 0, 0);
	L1.SetPartnerNumbers(0, 0, 1, 0);
	L2.SetPartnerNumbers(0, 0, 0, 1);
	S11.SetPartnerNumbers(2, 0, 0, 0);
	S12.SetPartnerNumbers(1, 1, 0, 0);
	S22.SetPartnerNumbers(0, 2, 0, 0);
	L11.SetPartnerNumbers(1, 0, 1, 0);
	L12.SetPartnerNumbers(0, 1, 1, 0);
	L21.SetPartnerNumbers(1, 0, 0, 1);
	L22.SetPartnerNumbers(0, 1, 0, 1);

	Virgin.VirginInd = 1;
}

void SexCohortM::Reset()
{
	Virgin.Reset();
	NoPartner.Reset();
	S1.Reset();
	S2.Reset();
	L1.Reset();
	L2.Reset();
	if(RiskGroup==1){
		S11.Reset();
		S12.Reset();
		S22.Reset();
		L11.Reset();
		L12.Reset();
		L21.Reset();
		L22.Reset();
	}
}

void SexCohortM::GetAllNumbersBySTDstage()
{
	Virgin.GetAllNumbersBySTDstage();
	NoPartner.GetAllNumbersBySTDstage();
	S1.GetAllNumbersBySTDstage();
	S2.GetAllNumbersBySTDstage();
	L1.GetAllNumbersBySTDstage();
	L2.GetAllNumbersBySTDstage();
	if(RiskGroup==1){
		S11.GetAllNumbersBySTDstage();
		S12.GetAllNumbersBySTDstage();
		S22.GetAllNumbersBySTDstage();
		L11.GetAllNumbersBySTDstage();
		L12.GetAllNumbersBySTDstage();
		L21.GetAllNumbersBySTDstage();
		L22.GetAllNumbersBySTDstage();
	}
}

void SexCohortM::GetAllPropnsBySTDstage()
{
	Virgin.GetAllPropnsBySTDstage();
	NoPartner.GetAllPropnsBySTDstage();
	S1.GetAllPropnsBySTDstage();
	S2.GetAllPropnsBySTDstage();
	L1.GetAllPropnsBySTDstage();
	L2.GetAllPropnsBySTDstage();
	if(RiskGroup==1){
		S11.GetAllPropnsBySTDstage();
		S12.GetAllPropnsBySTDstage();
		S22.GetAllPropnsBySTDstage();
		L11.GetAllPropnsBySTDstage();
		L12.GetAllPropnsBySTDstage();
		L21.GetAllPropnsBySTDstage();
		L22.GetAllPropnsBySTDstage();
	}
}

void SexCohortM::UpdateSyndromePropns()
{
	NoPartner.UpdateSyndromePropns();
	S1.UpdateSyndromePropns();
	S2.UpdateSyndromePropns();
	L1.UpdateSyndromePropns();
	L2.UpdateSyndromePropns();
	if(RiskGroup==1){
		S11.UpdateSyndromePropns();
		S12.UpdateSyndromePropns();
		S22.UpdateSyndromePropns();
		L11.UpdateSyndromePropns();
		L12.UpdateSyndromePropns();
		L21.UpdateSyndromePropns();
		L22.UpdateSyndromePropns();
	}
}

void SexCohortM::GetAllSTDcofactors()
{
	S1.GetAllSTDcofactors();
	S2.GetAllSTDcofactors();
	L1.GetAllSTDcofactors();
	L2.GetAllSTDcofactors();
	if(RiskGroup==1){
		NoPartner.GetAllSTDcofactors();
		S11.GetAllSTDcofactors();
		S12.GetAllSTDcofactors();
		S22.GetAllSTDcofactors();
		L11.GetAllSTDcofactors();
		L12.GetAllSTDcofactors();
		L21.GetAllSTDcofactors();
		L22.GetAllSTDcofactors();
	}
}

void SexCohortM::CalcTransmissionProb()
{
	if(RiskGroup==1){
		TPtransitionM.InfectProbFSW[0] = 0;
		HSVtransitionM.InfectProbFSW[0] = 0;
		HDtransitionM.InfectProbFSW[0] = 0;
		NGtransitionM.InfectProbFSW[0] = 0;
		CTtransitionM.InfectProbFSW[0] = 0;
		TVtransitionM.InfectProbFSW[0] = 0;
		HIVtransitionM.InfectProbFSW[0] = 0;
	}

	S1.CalcTransmissionProb();
	S2.CalcTransmissionProb();
	L1.CalcTransmissionProb();
	L2.CalcTransmissionProb();
	if(RiskGroup==1){
		NoPartner.CalcTransmissionProb();
		S11.CalcTransmissionProb();
		S12.CalcTransmissionProb();
		S22.CalcTransmissionProb();
		L11.CalcTransmissionProb();
		L12.CalcTransmissionProb();
		L21.CalcTransmissionProb();
		L22.CalcTransmissionProb();
	}
}

void SexCohortM::CalcInfectionProb()
{
	S1.CalcInfectionProb();
	S2.CalcInfectionProb();
	L1.CalcInfectionProb();
	L2.CalcInfectionProb();
	if(RiskGroup==1){
		NoPartner.CalcInfectionProb();
		S11.CalcInfectionProb();
		S12.CalcInfectionProb();
		S22.CalcInfectionProb();
		L11.CalcInfectionProb();
		L12.CalcInfectionProb();
		L21.CalcInfectionProb();
		L22.CalcInfectionProb();
	}
}

void SexCohortM::CalcSTDtransitions()
{
	NoPartner.CalcSTDtransitions();
	S1.CalcSTDtransitions();
	S2.CalcSTDtransitions();
	L1.CalcSTDtransitions();
	L2.CalcSTDtransitions();
	if(RiskGroup==1){
		S11.CalcSTDtransitions();
		S12.CalcSTDtransitions();
		S22.CalcSTDtransitions();
		L11.CalcSTDtransitions();
		L12.CalcSTDtransitions();
		L21.CalcSTDtransitions();
		L22.CalcSTDtransitions();
	}
}

void SexCohortM::HIVstageChanges()
{
	NoPartner.AllHIVstageChanges();
	S1.AllHIVstageChanges();
	S2.AllHIVstageChanges();
	L1.AllHIVstageChanges();
	L2.AllHIVstageChanges();
	if(RiskGroup==1){
		S11.AllHIVstageChanges();
		S12.AllHIVstageChanges();
		S22.AllHIVstageChanges();
		L11.AllHIVstageChanges();
		L12.AllHIVstageChanges();
		L21.AllHIVstageChanges();
		L22.AllHIVstageChanges();
	}
}

void SexCohortM::SetHIVnumbersToTemp()
{
	NoPartner.SetHIVnumbersToTemp();
	S1.SetHIVnumbersToTemp();
	S2.SetHIVnumbersToTemp();
	L1.SetHIVnumbersToTemp();
	L2.SetHIVnumbersToTemp();
	if(RiskGroup==1){
		S11.SetHIVnumbersToTemp();
		S12.SetHIVnumbersToTemp();
		S22.SetHIVnumbersToTemp();
		L11.SetHIVnumbersToTemp();
		L12.SetHIVnumbersToTemp();
		L21.SetHIVnumbersToTemp();
		L22.SetHIVnumbersToTemp();
	}
}

void SexCohortM::GetNewPartners()
{
	Virgin.GetNewPartners();
	NoPartner.GetNewPartners();
	S1.GetNewPartners();
	S2.GetNewPartners();
	if(RiskGroup==1){
		L1.GetNewPartners();
		L2.GetNewPartners();
		S11.GetNewPartners();
		S12.GetNewPartners();
		S22.GetNewPartners();
	}
}

void SexCohortM::GetPartnerTransitions()
{
	Virgin.GetPartnerTransitions();
	NoPartner.GetPartnerTransitions();
	S1.GetPartnerTransitions();
	S2.GetPartnerTransitions();
	L1.GetPartnerTransitions();
	L2.GetPartnerTransitions();
	if(RiskGroup==1){
		S11.GetPartnerTransitions();
		S12.GetPartnerTransitions();
		S22.GetPartnerTransitions();
		L11.GetPartnerTransitions();
		L12.GetPartnerTransitions();
		L21.GetPartnerTransitions();
		L22.GetPartnerTransitions();
	}
}

void SexCohortM::GetAllNumbersRemaining()
{
	Virgin.GetNumbersRemaining();
	NoPartner.GetNumbersRemaining();
	S1.GetNumbersRemaining();
	S2.GetNumbersRemaining();
	L1.GetNumbersRemaining();
	L2.GetNumbersRemaining();
	if(RiskGroup==1){
		S11.GetNumbersRemaining();
		S12.GetNumbersRemaining();
		S22.GetNumbersRemaining();
		L11.GetNumbersRemaining();
		L12.GetNumbersRemaining();
		L21.GetNumbersRemaining();
		L22.GetNumbersRemaining();
	}
}

void SexCohortM::GetNumbersChanging(SexuallyExpM* a, SexuallyExpM* b, double RatesDep[16][6])
{
	int ia, iz;
	double exits;

	for(ia=0; ia<16; ia++){
		exits = a->NumbersByHIVstage[ia][0] * RatesDep[ia][0];
		b->TempNumbersByHIVstage[ia][0] += exits;
		if(HSVind==1){
			for(iz=0; iz<HSVtransitionM.nStates; iz++){
				b->MHSV.NumberByStage0[ia][iz] += exits * a->MHSV.PropnByStage0[ia][iz];}
		}
		if(TPind==1){
			for(iz=0; iz<TPtransitionM.nStates; iz++){
				b->MTP.NumberByStage0[ia][iz] += exits * a->MTP.PropnByStage0[ia][iz];}
		}
		if(HDind==1){
			for(iz=0; iz<HDtransitionM.nStates; iz++){
				b->MHD.NumberByStage0[ia][iz] += exits * a->MHD.PropnByStage0[ia][iz];}
		}
		if(NGind==1){
			for(iz=0; iz<NGtransitionM.nStates; iz++){
				b->MNG.NumberByStage0[ia][iz] += exits * a->MNG.PropnByStage0[ia][iz];}
		}
		if(CTind==1){
			for(iz=0; iz<CTtransitionM.nStates; iz++){
				b->MCT.NumberByStage0[ia][iz] += exits * a->MCT.PropnByStage0[ia][iz];}
		}
		if(TVind==1){
			for(iz=0; iz<TVtransitionM.nStates; iz++){
				b->MTV.NumberByStage0[ia][iz] += exits * a->MTV.PropnByStage0[ia][iz];}
		}
		if(HIVind==1){
			// Repeat the above code for HIV stage 1:
			exits = a->NumbersByHIVstage[ia][1] * RatesDep[ia][1];
			b->TempNumbersByHIVstage[ia][1] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionM.nStates; iz++){
					b->MHSV.NumberByStage1[ia][iz] += exits * a->MHSV.PropnByStage1[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionM.nStates; iz++){
					b->MTP.NumberByStage1[ia][iz] += exits * a->MTP.PropnByStage1[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionM.nStates; iz++){
					b->MHD.NumberByStage1[ia][iz] += exits * a->MHD.PropnByStage1[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionM.nStates; iz++){
					b->MNG.NumberByStage1[ia][iz] += exits * a->MNG.PropnByStage1[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionM.nStates; iz++){
					b->MCT.NumberByStage1[ia][iz] += exits * a->MCT.PropnByStage1[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionM.nStates; iz++){
					b->MTV.NumberByStage1[ia][iz] += exits * a->MTV.PropnByStage1[ia][iz];}
			}
			// Repeat the above code for HIV stage 2:
			exits = a->NumbersByHIVstage[ia][2] * RatesDep[ia][2];
			b->TempNumbersByHIVstage[ia][2] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionM.nStates; iz++){
					b->MHSV.NumberByStage2[ia][iz] += exits * a->MHSV.PropnByStage2[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionM.nStates; iz++){
					b->MTP.NumberByStage2[ia][iz] += exits * a->MTP.PropnByStage2[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionM.nStates; iz++){
					b->MHD.NumberByStage2[ia][iz] += exits * a->MHD.PropnByStage2[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionM.nStates; iz++){
					b->MNG.NumberByStage2[ia][iz] += exits * a->MNG.PropnByStage2[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionM.nStates; iz++){
					b->MCT.NumberByStage2[ia][iz] += exits * a->MCT.PropnByStage2[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionM.nStates; iz++){
					b->MTV.NumberByStage2[ia][iz] += exits * a->MTV.PropnByStage2[ia][iz];}
			}
			// Repeat the above code for HIV stage 3:
			exits = a->NumbersByHIVstage[ia][3] * RatesDep[ia][3];
			b->TempNumbersByHIVstage[ia][3] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionM.nStates; iz++){
					b->MHSV.NumberByStage3[ia][iz] += exits * a->MHSV.PropnByStage3[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionM.nStates; iz++){
					b->MTP.NumberByStage3[ia][iz] += exits * a->MTP.PropnByStage3[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionM.nStates; iz++){
					b->MHD.NumberByStage3[ia][iz] += exits * a->MHD.PropnByStage3[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionM.nStates; iz++){
					b->MNG.NumberByStage3[ia][iz] += exits * a->MNG.PropnByStage3[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionM.nStates; iz++){
					b->MCT.NumberByStage3[ia][iz] += exits * a->MCT.PropnByStage3[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionM.nStates; iz++){
					b->MTV.NumberByStage3[ia][iz] += exits * a->MTV.PropnByStage3[ia][iz];}
			}
			// Repeat the above code for HIV stage 4:
			exits = a->NumbersByHIVstage[ia][4] * RatesDep[ia][4];
			b->TempNumbersByHIVstage[ia][4] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionM.nStates; iz++){
					b->MHSV.NumberByStage4[ia][iz] += exits * a->MHSV.PropnByStage4[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionM.nStates; iz++){
					b->MTP.NumberByStage4[ia][iz] += exits * a->MTP.PropnByStage4[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionM.nStates; iz++){
					b->MHD.NumberByStage4[ia][iz] += exits * a->MHD.PropnByStage4[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionM.nStates; iz++){
					b->MNG.NumberByStage4[ia][iz] += exits * a->MNG.PropnByStage4[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionM.nStates; iz++){
					b->MCT.NumberByStage4[ia][iz] += exits * a->MCT.PropnByStage4[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionM.nStates; iz++){
					b->MTV.NumberByStage4[ia][iz] += exits * a->MTV.PropnByStage4[ia][iz];}
			}
			// Repeat the above code for HIV stage 5:
			exits = a->NumbersByHIVstage[ia][5] * RatesDep[ia][5];
			b->TempNumbersByHIVstage[ia][5] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionM.nStates; iz++){
					b->MHSV.NumberByStage5[ia][iz] += exits * a->MHSV.PropnByStage5[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionM.nStates; iz++){
					b->MTP.NumberByStage5[ia][iz] += exits * a->MTP.PropnByStage5[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionM.nStates; iz++){
					b->MHD.NumberByStage5[ia][iz] += exits * a->MHD.PropnByStage5[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionM.nStates; iz++){
					b->MNG.NumberByStage5[ia][iz] += exits * a->MNG.PropnByStage5[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionM.nStates; iz++){
					b->MCT.NumberByStage5[ia][iz] += exits * a->MCT.PropnByStage5[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionM.nStates; iz++){
					b->MTV.NumberByStage5[ia][iz] += exits * a->MTV.PropnByStage5[ia][iz];}
			}
		}
	}
}

void SexCohortM::GetAllNumbersChanging()
{
	GetNumbersChanging(&Virgin, &S1, Virgin.AcquireNewS1dep);
	GetNumbersChanging(&Virgin, &S2, Virgin.AcquireNewS2dep);
	GetNumbersChanging(&NoPartner, &S1, NoPartner.AcquireNewS1dep);
	GetNumbersChanging(&NoPartner, &S2, NoPartner.AcquireNewS2dep);
	GetNumbersChanging(&S1, &NoPartner, S1.LoseS1dep);
	GetNumbersChanging(&S1, &L1, S1.MarryL1dep);
	GetNumbersChanging(&S2, &NoPartner, S2.LoseS2dep);
	GetNumbersChanging(&S2, &L2, S2.MarryL2dep);
	GetNumbersChanging(&L1, &NoPartner, L1.LoseLdep);
	GetNumbersChanging(&L2, &NoPartner, L2.LoseLdep);

	if(RiskGroup==1){
		GetNumbersChanging(&S1, &S11, S1.AcquireNewS1dep);
		GetNumbersChanging(&S1, &S12, S1.AcquireNewS2dep);
		GetNumbersChanging(&S2, &S12, S2.AcquireNewS1dep);
		GetNumbersChanging(&S2, &S22, S2.AcquireNewS2dep);
		GetNumbersChanging(&L1, &L11, L1.AcquireNewS1dep);
		GetNumbersChanging(&L1, &L12, L1.AcquireNewS2dep);
		GetNumbersChanging(&L2, &L21, L2.AcquireNewS1dep);
		GetNumbersChanging(&L2, &L22, L2.AcquireNewS2dep);
		GetNumbersChanging(&S11, &S1, S11.LoseS1dep);
		GetNumbersChanging(&S11, &L11, S11.MarryL1dep);
		GetNumbersChanging(&S12, &S1, S12.LoseS2dep);
		GetNumbersChanging(&S12, &S2, S12.LoseS1dep);
		GetNumbersChanging(&S12, &L12, S12.MarryL1dep);
		GetNumbersChanging(&S12, &L21, S12.MarryL2dep);
		GetNumbersChanging(&S22, &S2, S22.LoseS2dep);
		GetNumbersChanging(&S22, &L22, S22.MarryL2dep);
		GetNumbersChanging(&L11, &L1, L11.LoseS1dep);
		GetNumbersChanging(&L11, &S1, L11.LoseLdep);
		GetNumbersChanging(&L12, &L1, L12.LoseS2dep);
		GetNumbersChanging(&L12, &S2, L12.LoseLdep);
		GetNumbersChanging(&L21, &L2, L21.LoseS1dep);
		GetNumbersChanging(&L21, &S1, L21.LoseLdep);
		GetNumbersChanging(&L22, &L2, L22.LoseS2dep);
		GetNumbersChanging(&L22, &S2, L22.LoseLdep);
	}
}

void SexCohortM::CalcAllAgeChanges()
{
	Virgin.CalcAllAgeChanges();
	NoPartner.CalcAllAgeChanges();
	S1.CalcAllAgeChanges();
	S2.CalcAllAgeChanges();
	L1.CalcAllAgeChanges();
	L2.CalcAllAgeChanges();
	if(RiskGroup==1){
		S11.CalcAllAgeChanges();
		S12.CalcAllAgeChanges();
		S22.CalcAllAgeChanges();
		L11.CalcAllAgeChanges();
		L12.CalcAllAgeChanges();
		L21.CalcAllAgeChanges();
		L22.CalcAllAgeChanges();
	}
}

void SexCohortM::GetNewVirgins()
{
	double NewVirgins;

	if(RiskGroup==1){
		NewVirgins = MaleChild.HIVneg[9] * HighPropnM * (1.0 - MaleChild.NonAIDSmort[9]);}
	else{
		NewVirgins = MaleChild.HIVneg[9] * (1.0 - HighPropnM) * (1.0 - MaleChild.NonAIDSmort[9]);}

	Virgin.NumbersByHIVstage[0][0] += NewVirgins;
	Virgin.TotalAlive[0] += NewVirgins;
	if(HSVind==1){Virgin.MHSV.NumberByStage0[0][0] += NewVirgins;}
	if(TPind==1){Virgin.MTP.NumberByStage0[0][0] += NewVirgins;}
	if(HDind==1){Virgin.MHD.NumberByStage0[0][0] += NewVirgins;}
	if(NGind==1){Virgin.MNG.NumberByStage0[0][0] += NewVirgins;}
	if(CTind==1){Virgin.MCT.NumberByStage0[0][0] += NewVirgins;}
	if(TVind==1){Virgin.MTV.NumberByStage0[0][0] += NewVirgins;}
}

void SexCohortM::GetTotalBySTDstage()
{
	Virgin.GetTotalBySTDstage();
	NoPartner.GetTotalBySTDstage();
	S1.GetTotalBySTDstage();
	S2.GetTotalBySTDstage();
	L1.GetTotalBySTDstage();
	L2.GetTotalBySTDstage();
	if(RiskGroup==1){
		S11.GetTotalBySTDstage();
		S12.GetTotalBySTDstage();
		S22.GetTotalBySTDstage();
		L11.GetTotalBySTDstage();
		L12.GetTotalBySTDstage();
		L21.GetTotalBySTDstage();
		L22.GetTotalBySTDstage();
	}
}

void SexCohortM::GetTotalGUD()
{
	TotalGUDcases = 0;
	TotalGUDcases += NoPartner.ReturnTotalGUD();
	TotalGUDcases += S1.ReturnTotalGUD();
	TotalGUDcases += S2.ReturnTotalGUD();
	TotalGUDcases += L1.ReturnTotalGUD();
	TotalGUDcases += L2.ReturnTotalGUD();
	if(RiskGroup==1){
		TotalGUDcases += S11.ReturnTotalGUD();
		TotalGUDcases += S12.ReturnTotalGUD();
		TotalGUDcases += S22.ReturnTotalGUD();
		TotalGUDcases += L11.ReturnTotalGUD();
		TotalGUDcases += L12.ReturnTotalGUD();
		TotalGUDcases += L21.ReturnTotalGUD();
		TotalGUDcases += L22.ReturnTotalGUD();
	}
}

SexCohortF::SexCohortF(int Risk)
{
	SexInd = 1;
	RiskGroup = Risk;

	Virgin.RiskGroup = Risk;
	FSW.RiskGroup = Risk;
	NoPartner.RiskGroup = Risk;
	S1.RiskGroup = Risk;
	S2.RiskGroup = Risk;
	L1.RiskGroup = Risk;
	L2.RiskGroup = Risk;
	S11.RiskGroup = Risk;
	S12.RiskGroup = Risk;
	S22.RiskGroup = Risk;
	L11.RiskGroup = Risk;
	L12.RiskGroup = Risk;
	L21.RiskGroup = Risk;
	L22.RiskGroup = Risk;

	Virgin.SetPartnerNumbers(0, 0, 0, 0);
	FSW.SetPartnerNumbers(0, 0, 0, 0);
	NoPartner.SetPartnerNumbers(0, 0, 0, 0);
	S1.SetPartnerNumbers(1, 0, 0, 0);
	S2.SetPartnerNumbers(0, 1, 0, 0);
	L1.SetPartnerNumbers(0, 0, 1, 0);
	L2.SetPartnerNumbers(0, 0, 0, 1);
	S11.SetPartnerNumbers(2, 0, 0, 0);
	S12.SetPartnerNumbers(1, 1, 0, 0);
	S22.SetPartnerNumbers(0, 2, 0, 0);
	L11.SetPartnerNumbers(1, 0, 1, 0);
	L12.SetPartnerNumbers(0, 1, 1, 0);
	L21.SetPartnerNumbers(1, 0, 0, 1);
	L22.SetPartnerNumbers(0, 1, 0, 1);
	
	// Change the default settings for virgins and FSWs
	Virgin.VirginInd = 1;
	FSW.FSWind = 1;
}

void SexCohortF::Reset()
{
	Virgin.Reset();
	NoPartner.Reset();
	S1.Reset();
	S2.Reset();
	L1.Reset();
	L2.Reset();
	if(RiskGroup==1){
		FSW.Reset();
		S11.Reset();
		S12.Reset();
		S22.Reset();
		L11.Reset();
		L12.Reset();
		L21.Reset();
		L22.Reset();
	}

	/*if(HIVind==1){
		for(ia=0; ia<16; ia++){
			FSW.NumbersByHIVstage[ia][0] = FSW.TotalAlive[ia] * (1.0 - InitHIVprevHigh);
			FSW.NumbersByHIVstage[ia][2] = FSW.TotalAlive[ia] * InitHIVprevHigh;
			for(iz=0; iz<7; iz++){
				FSW.FHSV.PropnByStage2[ia][iz] = FSW.FHSV.PropnByStage0[ia][iz];
				FSW.FTP.PropnByStage2[ia][iz] = FSW.FTP.PropnByStage0[ia][iz];
				FSW.FHD.PropnByStage2[ia][iz] = FSW.FHD.PropnByStage0[ia][iz];
				FSW.FNG.PropnByStage2[ia][iz] = FSW.FNG.PropnByStage0[ia][iz];
				FSW.FCT.PropnByStage2[ia][iz] = FSW.FCT.PropnByStage0[ia][iz];
				FSW.FTV.PropnByStage2[ia][iz] = FSW.FTV.PropnByStage0[ia][iz];
				FSW.FBV.PropnByStage2[ia][iz] = FSW.FBV.PropnByStage0[ia][iz];
				FSW.FVC.PropnByStage2[ia][iz] = FSW.FVC.PropnByStage0[ia][iz];
			}
		}
	}*/
}

void SexCohortF::GetAllNumbersBySTDstage()
{
	Virgin.GetAllNumbersBySTDstage();
	NoPartner.GetAllNumbersBySTDstage();
	S1.GetAllNumbersBySTDstage();
	S2.GetAllNumbersBySTDstage();
	L1.GetAllNumbersBySTDstage();
	L2.GetAllNumbersBySTDstage();
	if(RiskGroup==1){
		FSW.GetAllNumbersBySTDstage();
		S11.GetAllNumbersBySTDstage();
		S12.GetAllNumbersBySTDstage();
		S22.GetAllNumbersBySTDstage();
		L11.GetAllNumbersBySTDstage();
		L12.GetAllNumbersBySTDstage();
		L21.GetAllNumbersBySTDstage();
		L22.GetAllNumbersBySTDstage();
	}
}

void SexCohortF::GetAllPropnsBySTDstage()
{
	Virgin.GetAllPropnsBySTDstage();
	NoPartner.GetAllPropnsBySTDstage();
	S1.GetAllPropnsBySTDstage();
	S2.GetAllPropnsBySTDstage();
	L1.GetAllPropnsBySTDstage();
	L2.GetAllPropnsBySTDstage();
	if(RiskGroup==1){
		FSW.GetAllPropnsBySTDstage();
		S11.GetAllPropnsBySTDstage();
		S12.GetAllPropnsBySTDstage();
		S22.GetAllPropnsBySTDstage();
		L11.GetAllPropnsBySTDstage();
		L12.GetAllPropnsBySTDstage();
		L21.GetAllPropnsBySTDstage();
		L22.GetAllPropnsBySTDstage();
	}
}

void SexCohortF::UpdateSyndromePropns()
{
	NoPartner.UpdateSyndromePropns();
	S1.UpdateSyndromePropns();
	S2.UpdateSyndromePropns();
	L1.UpdateSyndromePropns();
	L2.UpdateSyndromePropns();
	if(RiskGroup==1){
		FSW.UpdateSyndromePropns();
		S11.UpdateSyndromePropns();
		S12.UpdateSyndromePropns();
		S22.UpdateSyndromePropns();
		L11.UpdateSyndromePropns();
		L12.UpdateSyndromePropns();
		L21.UpdateSyndromePropns();
		L22.UpdateSyndromePropns();
	}
}

void SexCohortF::GetAllSTDcofactors()
{
	S1.GetAllSTDcofactors();
	S2.GetAllSTDcofactors();
	L1.GetAllSTDcofactors();
	L2.GetAllSTDcofactors();
	if(RiskGroup==1){
		FSW.GetAllSTDcofactors();
		S11.GetAllSTDcofactors();
		S12.GetAllSTDcofactors();
		S22.GetAllSTDcofactors();
		L11.GetAllSTDcofactors();
		L12.GetAllSTDcofactors();
		L21.GetAllSTDcofactors();
		L22.GetAllSTDcofactors();
	}
}

void SexCohortF::CalcTransmissionProb()
{
	if(RiskGroup==1){
		TPtransitionF.InfectProbFSW[0] = 0;
		HSVtransitionF.InfectProbFSW[0] = 0;
		HDtransitionF.InfectProbFSW[0] = 0;
		NGtransitionF.InfectProbFSW[0] = 0;
		CTtransitionF.InfectProbFSW[0] = 0;
		TVtransitionF.InfectProbFSW[0] = 0;
		HIVtransitionF.InfectProbFSW[0] = 0;
	}

	S1.CalcTransmissionProb();
	S2.CalcTransmissionProb();
	L1.CalcTransmissionProb();
	L2.CalcTransmissionProb();
	if(RiskGroup==1){
		FSW.CalcTransmissionProb();
		S11.CalcTransmissionProb();
		S12.CalcTransmissionProb();
		S22.CalcTransmissionProb();
		L11.CalcTransmissionProb();
		L12.CalcTransmissionProb();
		L21.CalcTransmissionProb();
		L22.CalcTransmissionProb();
	}
}

void SexCohortF::CalcInfectionProb()
{
	S1.CalcInfectionProb();
	S2.CalcInfectionProb();
	L1.CalcInfectionProb();
	L2.CalcInfectionProb();
	if(RiskGroup==1){
		FSW.CalcInfectionProb();
		S11.CalcInfectionProb();
		S12.CalcInfectionProb();
		S22.CalcInfectionProb();
		L11.CalcInfectionProb();
		L12.CalcInfectionProb();
		L21.CalcInfectionProb();
		L22.CalcInfectionProb();
	}
}

void SexCohortF::CalcSTDtransitions()
{
	Virgin.CalcSTDtransitions();
	NoPartner.CalcSTDtransitions();
	S1.CalcSTDtransitions();
	S2.CalcSTDtransitions();
	L1.CalcSTDtransitions();
	L2.CalcSTDtransitions();
	if(RiskGroup==1){
		FSW.CalcSTDtransitions();
		S11.CalcSTDtransitions();
		S12.CalcSTDtransitions();
		S22.CalcSTDtransitions();
		L11.CalcSTDtransitions();
		L12.CalcSTDtransitions();
		L21.CalcSTDtransitions();
		L22.CalcSTDtransitions();
	}
}

void SexCohortF::HIVstageChanges()
{
	int ia, iz;

	// Set Propn = TempPropn for the BV and VC objects in the virgin group
	for(ia=0; ia<4; ia++){
		for(iz=0; iz<3; iz++){
			Virgin.FVC.PropnByStage0[ia][iz] = Virgin.FVC.TempPropnByStage0[ia][iz];}
		for(iz=0; iz<4; iz++){
			Virgin.FBV.PropnByStage0[ia][iz] = Virgin.FBV.TempPropnByStage0[ia][iz];}
	}

	NoPartner.AllHIVstageChanges();
	S1.AllHIVstageChanges();
	S2.AllHIVstageChanges();
	L1.AllHIVstageChanges();
	L2.AllHIVstageChanges();
	if(RiskGroup==1){
		FSW.AllHIVstageChanges();
		S11.AllHIVstageChanges();
		S12.AllHIVstageChanges();
		S22.AllHIVstageChanges();
		L11.AllHIVstageChanges();
		L12.AllHIVstageChanges();
		L21.AllHIVstageChanges();
		L22.AllHIVstageChanges();
	}
}

void SexCohortF::SetHIVnumbersToTemp()
{
	NoPartner.SetHIVnumbersToTemp();
	S1.SetHIVnumbersToTemp();
	S2.SetHIVnumbersToTemp();
	L1.SetHIVnumbersToTemp();
	L2.SetHIVnumbersToTemp();
	if(RiskGroup==1){
		FSW.SetHIVnumbersToTemp();
		S11.SetHIVnumbersToTemp();
		S12.SetHIVnumbersToTemp();
		S22.SetHIVnumbersToTemp();
		L11.SetHIVnumbersToTemp();
		L12.SetHIVnumbersToTemp();
		L21.SetHIVnumbersToTemp();
		L22.SetHIVnumbersToTemp();
	}
}

void SexCohortF::GetNewPartners()
{
	Virgin.GetNewPartners();
	NoPartner.GetNewPartners();
	S1.GetNewPartners();
	S2.GetNewPartners();
	if(RiskGroup==1){
		L1.GetNewPartners();
		L2.GetNewPartners();
		S11.GetNewPartners();
		S12.GetNewPartners();
		S22.GetNewPartners();
	}
}

void SexCohortF::GetPartnerTransitions()
{
	Virgin.GetPartnerTransitions();
	NoPartner.GetPartnerTransitions();
	S1.GetPartnerTransitions();
	S2.GetPartnerTransitions();
	L1.GetPartnerTransitions();
	L2.GetPartnerTransitions();
	if(RiskGroup==1){
		FSW.GetPartnerTransitions();
		S11.GetPartnerTransitions();
		S12.GetPartnerTransitions();
		S22.GetPartnerTransitions();
		L11.GetPartnerTransitions();
		L12.GetPartnerTransitions();
		L21.GetPartnerTransitions();
		L22.GetPartnerTransitions();
	}
}

void SexCohortF::GetAllNumbersRemaining()
{
	Virgin.GetNumbersRemaining();
	NoPartner.GetNumbersRemaining();
	S1.GetNumbersRemaining();
	S2.GetNumbersRemaining();
	L1.GetNumbersRemaining();
	L2.GetNumbersRemaining();
	if(RiskGroup==1){
		FSW.GetNumbersRemaining();
		S11.GetNumbersRemaining();
		S12.GetNumbersRemaining();
		S22.GetNumbersRemaining();
		L11.GetNumbersRemaining();
		L12.GetNumbersRemaining();
		L21.GetNumbersRemaining();
		L22.GetNumbersRemaining();
	}
}

void SexCohortF::GetNumbersChanging(SexuallyExpF* a, SexuallyExpF* b, double RatesDep[16][6])
{
	int ia, iz;
	double exits;

	for(ia=0; ia<16; ia++){
		exits = a->NumbersByHIVstage[ia][0] * RatesDep[ia][0];
		b->TempNumbersByHIVstage[ia][0] += exits;
		if(HSVind==1){
			for(iz=0; iz<HSVtransitionF.nStates; iz++){
				b->FHSV.NumberByStage0[ia][iz] += exits * a->FHSV.PropnByStage0[ia][iz];}
		}
		if(TPind==1){
			for(iz=0; iz<TPtransitionF.nStates; iz++){
				b->FTP.NumberByStage0[ia][iz] += exits * a->FTP.PropnByStage0[ia][iz];}
		}
		if(HDind==1){
			for(iz=0; iz<HDtransitionF.nStates; iz++){
				b->FHD.NumberByStage0[ia][iz] += exits * a->FHD.PropnByStage0[ia][iz];}
		}
		if(NGind==1){
			for(iz=0; iz<NGtransitionF.nStates; iz++){
				b->FNG.NumberByStage0[ia][iz] += exits * a->FNG.PropnByStage0[ia][iz];}
		}
		if(CTind==1){
			for(iz=0; iz<CTtransitionF.nStates; iz++){
				b->FCT.NumberByStage0[ia][iz] += exits * a->FCT.PropnByStage0[ia][iz];}
		}
		if(TVind==1){
			for(iz=0; iz<TVtransitionF.nStates; iz++){
				b->FTV.NumberByStage0[ia][iz] += exits * a->FTV.PropnByStage0[ia][iz];}
		}
		if(VCind==1){
			for(iz=0; iz<VCtransitionF.nStates; iz++){
				b->FVC.NumberByStage0[ia][iz] += exits * a->FVC.PropnByStage0[ia][iz];}
		}
		if(BVind==1){
			for(iz=0; iz<BVtransitionF.nStates; iz++){
				b->FBV.NumberByStage0[ia][iz] += exits * a->FBV.PropnByStage0[ia][iz];}
		}
		if(HIVind==1){
			// Repeat the above code for HIV stage 1
			exits = a->NumbersByHIVstage[ia][1] * RatesDep[ia][1];
			b->TempNumbersByHIVstage[ia][1] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionF.nStates; iz++){
					b->FHSV.NumberByStage1[ia][iz] += exits * a->FHSV.PropnByStage1[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionF.nStates; iz++){
					b->FTP.NumberByStage1[ia][iz] += exits * a->FTP.PropnByStage1[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionF.nStates; iz++){
					b->FHD.NumberByStage1[ia][iz] += exits * a->FHD.PropnByStage1[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionF.nStates; iz++){
					b->FNG.NumberByStage1[ia][iz] += exits * a->FNG.PropnByStage1[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionF.nStates; iz++){
					b->FCT.NumberByStage1[ia][iz] += exits * a->FCT.PropnByStage1[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionF.nStates; iz++){
					b->FTV.NumberByStage1[ia][iz] += exits * a->FTV.PropnByStage1[ia][iz];}
			}
			if(VCind==1){
				for(iz=0; iz<VCtransitionF.nStates; iz++){
					b->FVC.NumberByStage1[ia][iz] += exits * a->FVC.PropnByStage1[ia][iz];}
			}
			if(BVind==1){
				for(iz=0; iz<BVtransitionF.nStates; iz++){
					b->FBV.NumberByStage1[ia][iz] += exits * a->FBV.PropnByStage1[ia][iz];}
			}
			// Repeat the above code for HIV stage 2
			exits = a->NumbersByHIVstage[ia][2] * RatesDep[ia][2];
			b->TempNumbersByHIVstage[ia][2] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionF.nStates; iz++){
					b->FHSV.NumberByStage2[ia][iz] += exits * a->FHSV.PropnByStage2[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionF.nStates; iz++){
					b->FTP.NumberByStage2[ia][iz] += exits * a->FTP.PropnByStage2[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionF.nStates; iz++){
					b->FHD.NumberByStage2[ia][iz] += exits * a->FHD.PropnByStage2[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionF.nStates; iz++){
					b->FNG.NumberByStage2[ia][iz] += exits * a->FNG.PropnByStage2[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionF.nStates; iz++){
					b->FCT.NumberByStage2[ia][iz] += exits * a->FCT.PropnByStage2[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionF.nStates; iz++){
					b->FTV.NumberByStage2[ia][iz] += exits * a->FTV.PropnByStage2[ia][iz];}
			}
			if(VCind==1){
				for(iz=0; iz<VCtransitionF.nStates; iz++){
					b->FVC.NumberByStage2[ia][iz] += exits * a->FVC.PropnByStage2[ia][iz];}
			}
			if(BVind==1){
				for(iz=0; iz<BVtransitionF.nStates; iz++){
					b->FBV.NumberByStage2[ia][iz] += exits * a->FBV.PropnByStage2[ia][iz];}
			}
			// Repeat the above code for HIV stage 3
			exits = a->NumbersByHIVstage[ia][3] * RatesDep[ia][3];
			b->TempNumbersByHIVstage[ia][3] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionF.nStates; iz++){
					b->FHSV.NumberByStage3[ia][iz] += exits * a->FHSV.PropnByStage3[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionF.nStates; iz++){
					b->FTP.NumberByStage3[ia][iz] += exits * a->FTP.PropnByStage3[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionF.nStates; iz++){
					b->FHD.NumberByStage3[ia][iz] += exits * a->FHD.PropnByStage3[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionF.nStates; iz++){
					b->FNG.NumberByStage3[ia][iz] += exits * a->FNG.PropnByStage3[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionF.nStates; iz++){
					b->FCT.NumberByStage3[ia][iz] += exits * a->FCT.PropnByStage3[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionF.nStates; iz++){
					b->FTV.NumberByStage3[ia][iz] += exits * a->FTV.PropnByStage3[ia][iz];}
			}
			if(VCind==1){
				for(iz=0; iz<VCtransitionF.nStates; iz++){
					b->FVC.NumberByStage3[ia][iz] += exits * a->FVC.PropnByStage3[ia][iz];}
			}
			if(BVind==1){
				for(iz=0; iz<BVtransitionF.nStates; iz++){
					b->FBV.NumberByStage3[ia][iz] += exits * a->FBV.PropnByStage3[ia][iz];}
			}
			// Repeat the above code for HIV stage 4
			exits = a->NumbersByHIVstage[ia][4] * RatesDep[ia][4];
			b->TempNumbersByHIVstage[ia][4] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionF.nStates; iz++){
					b->FHSV.NumberByStage4[ia][iz] += exits * a->FHSV.PropnByStage4[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionF.nStates; iz++){
					b->FTP.NumberByStage4[ia][iz] += exits * a->FTP.PropnByStage4[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionF.nStates; iz++){
					b->FHD.NumberByStage4[ia][iz] += exits * a->FHD.PropnByStage4[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionF.nStates; iz++){
					b->FNG.NumberByStage4[ia][iz] += exits * a->FNG.PropnByStage4[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionF.nStates; iz++){
					b->FCT.NumberByStage4[ia][iz] += exits * a->FCT.PropnByStage4[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionF.nStates; iz++){
					b->FTV.NumberByStage4[ia][iz] += exits * a->FTV.PropnByStage4[ia][iz];}
			}
			if(VCind==1){
				for(iz=0; iz<VCtransitionF.nStates; iz++){
					b->FVC.NumberByStage4[ia][iz] += exits * a->FVC.PropnByStage4[ia][iz];}
			}
			if(BVind==1){
				for(iz=0; iz<BVtransitionF.nStates; iz++){
					b->FBV.NumberByStage4[ia][iz] += exits * a->FBV.PropnByStage4[ia][iz];}
			}
			// Repeat the above code for HIV stage 5
			exits = a->NumbersByHIVstage[ia][5] * RatesDep[ia][5];
			b->TempNumbersByHIVstage[ia][5] += exits;
			if(HSVind==1){
				for(iz=0; iz<HSVtransitionF.nStates; iz++){
					b->FHSV.NumberByStage5[ia][iz] += exits * a->FHSV.PropnByStage5[ia][iz];}
			}
			if(TPind==1){
				for(iz=0; iz<TPtransitionF.nStates; iz++){
					b->FTP.NumberByStage5[ia][iz] += exits * a->FTP.PropnByStage5[ia][iz];}
			}
			if(HDind==1){
				for(iz=0; iz<HDtransitionF.nStates; iz++){
					b->FHD.NumberByStage5[ia][iz] += exits * a->FHD.PropnByStage5[ia][iz];}
			}
			if(NGind==1){
				for(iz=0; iz<NGtransitionF.nStates; iz++){
					b->FNG.NumberByStage5[ia][iz] += exits * a->FNG.PropnByStage5[ia][iz];}
			}
			if(CTind==1){
				for(iz=0; iz<CTtransitionF.nStates; iz++){
					b->FCT.NumberByStage5[ia][iz] += exits * a->FCT.PropnByStage5[ia][iz];}
			}
			if(TVind==1){
				for(iz=0; iz<TVtransitionF.nStates; iz++){
					b->FTV.NumberByStage5[ia][iz] += exits * a->FTV.PropnByStage5[ia][iz];}
			}
			if(VCind==1){
				for(iz=0; iz<VCtransitionF.nStates; iz++){
					b->FVC.NumberByStage5[ia][iz] += exits * a->FVC.PropnByStage5[ia][iz];}
			}
			if(BVind==1){
				for(iz=0; iz<BVtransitionF.nStates; iz++){
					b->FBV.NumberByStage5[ia][iz] += exits * a->FBV.PropnByStage5[ia][iz];}
			}
		}
	}
}

void SexCohortF::GetAllNumbersChanging()
{
	GetNumbersChanging(&Virgin, &S1, Virgin.AcquireNewS1dep);
	GetNumbersChanging(&Virgin, &S2, Virgin.AcquireNewS2dep);
	GetNumbersChanging(&NoPartner, &S1, NoPartner.AcquireNewS1dep);
	GetNumbersChanging(&NoPartner, &S2, NoPartner.AcquireNewS2dep);
	GetNumbersChanging(&S1, &NoPartner, S1.LoseS1dep);
	GetNumbersChanging(&S1, &L1, S1.MarryL1dep);
	GetNumbersChanging(&S2, &NoPartner, S2.LoseS2dep);
	GetNumbersChanging(&S2, &L2, S2.MarryL2dep);
	GetNumbersChanging(&L1, &NoPartner, L1.LoseLdep);
	GetNumbersChanging(&L2, &NoPartner, L2.LoseLdep);

	if(RiskGroup==1){
		GetNumbersChanging(&FSW, &NoPartner, FSW.LeaveSWdep);
		GetNumbersChanging(&NoPartner, &FSW, NoPartner.EnterSWdep);
		GetNumbersChanging(&S1, &S11, S1.AcquireNewS1dep);
		GetNumbersChanging(&S1, &S12, S1.AcquireNewS2dep);
		GetNumbersChanging(&S2, &S12, S2.AcquireNewS1dep);
		GetNumbersChanging(&S2, &S22, S2.AcquireNewS2dep);
		GetNumbersChanging(&L1, &L11, L1.AcquireNewS1dep);
		GetNumbersChanging(&L1, &L12, L1.AcquireNewS2dep);
		GetNumbersChanging(&L2, &L21, L2.AcquireNewS1dep);
		GetNumbersChanging(&L2, &L22, L2.AcquireNewS2dep);
		GetNumbersChanging(&S11, &S1, S11.LoseS1dep);
		GetNumbersChanging(&S11, &L11, S11.MarryL1dep);
		GetNumbersChanging(&S12, &S1, S12.LoseS2dep);
		GetNumbersChanging(&S12, &S2, S12.LoseS1dep);
		GetNumbersChanging(&S12, &L12, S12.MarryL1dep);
		GetNumbersChanging(&S12, &L21, S12.MarryL2dep);
		GetNumbersChanging(&S22, &S2, S22.LoseS2dep);
		GetNumbersChanging(&S22, &L22, S22.MarryL2dep);
		GetNumbersChanging(&L11, &L1, L11.LoseS1dep);
		GetNumbersChanging(&L11, &S1, L11.LoseLdep);
		GetNumbersChanging(&L12, &L1, L12.LoseS2dep);
		GetNumbersChanging(&L12, &S2, L12.LoseLdep);
		GetNumbersChanging(&L21, &L2, L21.LoseS1dep);
		GetNumbersChanging(&L21, &S1, L21.LoseLdep);
		GetNumbersChanging(&L22, &L2, L22.LoseS2dep);
		GetNumbersChanging(&L22, &S2, L22.LoseLdep);
	}
}

void SexCohortF::CalcAllAgeChanges()
{
	Virgin.CalcAllAgeChanges();
	NoPartner.CalcAllAgeChanges();
	S1.CalcAllAgeChanges();
	S2.CalcAllAgeChanges();
	L1.CalcAllAgeChanges();
	L2.CalcAllAgeChanges();
	if(RiskGroup==1){
		FSW.CalcAllAgeChanges();
		S11.CalcAllAgeChanges();
		S12.CalcAllAgeChanges();
		S22.CalcAllAgeChanges();
		L11.CalcAllAgeChanges();
		L12.CalcAllAgeChanges();
		L21.CalcAllAgeChanges();
		L22.CalcAllAgeChanges();
	}
}

void SexCohortF::GetNewVirgins()
{
	double NewVirgins;

	if(RiskGroup==1){
		NewVirgins = FemChild.HIVneg[9] * HighPropnF * (1.0 - FemChild.NonAIDSmort[9]);}
	else{
		NewVirgins = FemChild.HIVneg[9] * (1.0 - HighPropnF) * (1.0 - FemChild.NonAIDSmort[9]);}

	Virgin.NumbersByHIVstage[0][0] += NewVirgins;
	Virgin.TotalAlive[0] += NewVirgins;
	if(HSVind==1){Virgin.FHSV.NumberByStage0[0][0] += NewVirgins;}
	if(TPind==1){Virgin.FTP.NumberByStage0[0][0] += NewVirgins;}
	if(HDind==1){Virgin.FHD.NumberByStage0[0][0] += NewVirgins;}
	if(NGind==1){Virgin.FNG.NumberByStage0[0][0] += NewVirgins;}
	if(CTind==1){Virgin.FCT.NumberByStage0[0][0] += NewVirgins;}
	if(TVind==1){Virgin.FTV.NumberByStage0[0][0] += NewVirgins;}
	if(VCind==1){Virgin.FVC.NumberByStage0[0][0] += NewVirgins;}
	if(BVind==1){Virgin.FBV.NumberByStage0[0][0] += NewVirgins;}
}

void SexCohortF::GetTotalBySTDstage()
{
	Virgin.GetTotalBySTDstage();
	NoPartner.GetTotalBySTDstage();
	S1.GetTotalBySTDstage();
	S2.GetTotalBySTDstage();
	L1.GetTotalBySTDstage();
	L2.GetTotalBySTDstage();
	if(RiskGroup==1){
		FSW.GetTotalBySTDstage();
		S11.GetTotalBySTDstage();
		S12.GetTotalBySTDstage();
		S22.GetTotalBySTDstage();
		L11.GetTotalBySTDstage();
		L12.GetTotalBySTDstage();
		L21.GetTotalBySTDstage();
		L22.GetTotalBySTDstage();
	}
}

void SexCohortF::GetTotalGUD()
{
	TotalGUDcases = 0;
	TotalGUDcases += NoPartner.ReturnTotalGUD();
	TotalGUDcases += S1.ReturnTotalGUD();
	TotalGUDcases += S2.ReturnTotalGUD();
	TotalGUDcases += L1.ReturnTotalGUD();
	TotalGUDcases += L2.ReturnTotalGUD();
	if(RiskGroup==1){
		TotalGUDcases += FSW.ReturnTotalGUD();
		TotalGUDcases += S11.ReturnTotalGUD();
		TotalGUDcases += S12.ReturnTotalGUD();
		TotalGUDcases += S22.ReturnTotalGUD();
		TotalGUDcases += L11.ReturnTotalGUD();
		TotalGUDcases += L12.ReturnTotalGUD();
		TotalGUDcases += L21.ReturnTotalGUD();
		TotalGUDcases += L22.ReturnTotalGUD();
	}
}

Child::Child(int Sex)
{
	SexInd = Sex;
}

void Child::Reset()
{
	int ia;

	for(ia=0; ia<15; ia++){
		Perinatal.PreAIDSstage[ia] = 0.0;
		Perinatal.AIDSstage[ia] = 0.0;
		Perinatal.AIDSdeaths[ia] = 0.0;
		Breastmilk.PreAIDSstage[ia] = 0.0;
		Breastmilk.AIDSstage[ia] = 0.0;
		Breastmilk.AIDSdeaths[ia] = 0.0;
		OnHAART[ia] = 0.0;
		AIDSdeathsTot[ia] = 0.0;
		if(ia<10){
			TotalAlive[ia] = HIVneg[ia];}
		else{
			TotalAlive[ia] = 0.0;}
	}
}

void Child::UpdateMort()
{
	int ia, yr;

	yr = CurrYear - StartYear;

	if(SexInd==0){
		MortProb1st6m = InfantMort1st6mM[yr];}
	else{
		MortProb1st6m = InfantMort1st6mF[yr];}
	Perinatal.MortProb1st6m = MortProb1st6m;
	Breastmilk.MortProb1st6m = MortProb1st6m;
	for(ia=0; ia<15; ia++){
		if(SexInd==0){
			NonAIDSmort[ia] = ChildMortM[ia][yr];}
		else{
			NonAIDSmort[ia] = ChildMortF[ia][yr];}
		Perinatal.NonAIDSmort[ia] = NonAIDSmort[ia];
		Breastmilk.NonAIDSmort[ia] = NonAIDSmort[ia];
	}
}

void Child::UpdateBirths()
{
	NegBirths = (TotalBirths - NewHIVperinatal - NewHIVbreastmilk) * PropnBirths;
	Perinatal.NewHIV = NewHIVperinatal * PropnBirths;
	Breastmilk.NewHIV = NewHIVbreastmilk * PropnBirths;
}

void Child::CalcAgeChanges()
{
	int ia, yr;
	double HAARTmort;

	yr = CurrYear - StartYear;
	if(SexInd==0){
		HAARTmort = 1.0/HIVtransitionM.AveDuration[4];}
	else{
		HAARTmort = 1.0/HIVtransitionF.AveDuration[4];}

	Perinatal.CalcAgeChanges();
	Breastmilk.CalcAgeChanges();

	// Calculate AIDS deaths
	AIDSdeathsTot[0] = (Perinatal.NewHIV * Perinatal.AIDSprob1st6m + Breastmilk.NewHIV
		* Breastmilk.AIDSprob1st6m) * HAARTaccess[yr] * (1.0 - exp(-0.25 * 52.0 * HAARTmort))
		* (1.0 - MortProb1st6m) + Perinatal.AIDSdeaths[0] + Breastmilk.AIDSdeaths[0];
	for(ia=1; ia<15; ia++){
		AIDSdeathsTot[ia] = (OnHAART[ia-1] * (1.0 - exp(-52.0 * HAARTmort)) 
			+ (Perinatal.PreAIDSstage[ia-1] * Perinatal.AIDSprogressionRate[ia-1] 
			+ Breastmilk.PreAIDSstage[ia-1] * Breastmilk.AIDSprogressionRate[ia-1]) 
			* HAARTaccess[yr] * (1.0 - exp(-0.5 * 52.0 * HAARTmort))) * (1.0 - 
			NonAIDSmort[ia-1]); 
	}

	// Calculate age changes in HIV-negative children
	for(ia=0; ia<9; ia++){
		HIVneg[9-ia] = HIVneg[8-ia] * (1.0 - NonAIDSmort[8-ia]);}
	HIVneg[0] = NegBirths * (1.0 - MortProb1st6m);

	// Calculate age changes in children on HAART + new HAART recipients
	for(ia=0; ia<14; ia++){
		OnHAART[14-ia] = (OnHAART[13-ia] * exp(-52.0 * HAARTmort) +
			(Perinatal.PreAIDSstage[13-ia] * Perinatal.AIDSprogressionRate[13-ia] +
			Breastmilk.PreAIDSstage[13-ia] * Breastmilk.AIDSprogressionRate[13-ia]) *
			HAARTaccess[yr] * exp(-0.5 * 52.0 * HAARTmort)) * (1.0 - NonAIDSmort[13-ia]);
	}
	OnHAART[0] = (Perinatal.NewHIV * Perinatal.AIDSprob1st6m + Breastmilk.NewHIV * 
		Breastmilk.AIDSprob1st6m) * HAARTaccess[yr] * exp(-0.25 * 52.0 * HAARTmort) * 
		(1.0 - MortProb1st6m);

	// Calculate total alive at end of year
	for(ia=0; ia<10; ia++){
		TotalAlive[ia] = HIVneg[ia] + Perinatal.PreAIDSstage[ia] + Perinatal.AIDSstage[ia] +
			Breastmilk.PreAIDSstage[ia] + Breastmilk.AIDSstage[ia] + OnHAART[ia];
	}
	for(ia=10; ia<15; ia++){
		TotalAlive[ia] = Perinatal.PreAIDSstage[ia] + Perinatal.AIDSstage[ia] +
			Breastmilk.PreAIDSstage[ia] + Breastmilk.AIDSstage[ia] + OnHAART[ia];
	}
}

void ReadSexAssumps(const char* input)
{
	int ia, ib, is;
	ifstream file;

	file.open(input);
	if(file==0)cout<<"File open error"<<endl;
	file.ignore(255,'\n');
	file>>HighPropnM>>HighPropnF;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>AssortativeM>>AssortativeF;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>GenderEquality;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>AnnNumberClients;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<4; ia++){
		file>>SexualDebut[ia][0]>>SexualDebut[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>DebutAdjLow[0]>>DebutAdjLow[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>PartnershipFormation[0][0]>>PartnershipFormation[1][0]>>
		PartnershipFormation[0][1]>>PartnershipFormation[1][1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>BasePartnerAcqH[0]>>BasePartnerAcqH[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		file>>AgeEffectPartners[ia][0];}
	for(ia=0; ia<16; ia++){
		file>>AgeEffectPartners[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>GammaMeanST[0]>>GammaMeanST[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>GammaStdDevST[0]>>GammaStdDevST[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>PartnerEffectNew[0][0]>>PartnerEffectNew[0][1]>>PartnerEffectNew[1][0]>>
		PartnerEffectNew[1][1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(is=0; is<5; is++){
		file>>HIVeffectPartners[is];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		for(ib=0; ib<2; ib++){
			file>>MarriageIncidence[ia][ib];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>MeanFSWcontacts>>GammaMeanFSW>>GammaStdDevFSW;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ib=0; ib<5; ib++){
		file>>PartnerEffectFSWcontact[ib];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		file>>InitFSWageDbn[ia]>>FSWexit[ia];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(is=0; is<5; is++){
		file>>HIVeffectFSWentry[is];}
	for(is=0; is<5; is++){
		file>>HIVeffectFSWexit[is];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>MeanDurSTrel[0][0]>>MeanDurSTrel[0][1]>>MeanDurSTrel[1][0]>>MeanDurSTrel[1][1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		file>>LTseparation[ia][0]>>LTseparation[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		for(ib=0; ib<16; ib++){
			file>>AgePrefF[ia][ib];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		for(ib=0; ib<16; ib++){
			file>>AgePrefM[ia][ib];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		file>>FreqSexST[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		file>>FreqSexLT[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>BaselineCondomSvy;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ib=0; ib<3; ib++){
		file>>RelEffectCondom[ib];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ib=0; ib<3; ib++){
		file>>AgeEffectCondom[ib];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ib=0; ib<3; ib++){
		file>>RatioInitialTo1998[ib];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ib=0; ib<3; ib++){
		file>>RatioUltTo1998[ib];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ib=0; ib<3; ib++){
		file>>MedianToBehavChange[ib];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>DebutBias[0][0]>>DebutBias[0][1]>>DebutBias[1][0]>>DebutBias[1][1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>AbstinenceBias[0]>>AbstinenceBias[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>ConcurrencyBias[0][0]>>ConcurrencyBias[0][1]>>ConcurrencyBias[1][0]>>
		ConcurrencyBias[1][1];
	file.close();

	// Calculate frequency of sex in men
	double sumxy;
	for(ia=0; ia<16; ia++){
		sumxy = 0;
		for(ib=0; ib<16; ib++){
			sumxy += AgePrefM[ia][ib] * FreqSexST[ib][1];}
		FreqSexST[ia][0] = sumxy;
	}
	for(ia=0; ia<16; ia++){
		sumxy = 0;
		for(ib=0; ib<16; ib++){
			sumxy += AgePrefM[ia][ib] * FreqSexLT[ib][1];}
		FreqSexLT[ia][0] = sumxy;
	}

	// Set BaselineCondomUse
	BaselineCondomUse = BaselineCondomSvy;

	// Calculate Weibull shape parameters for pace of behaviour change
	for(ib=0; ib<3; ib++){
		ShapeBehavChange[ib] = log(log(1.0 - log(RatioInitialTo1998[ib])/
			log(RatioUltTo1998[ib]))/log(2.0))/log(13.0/MedianToBehavChange[ib]);
	}

	// Replace the default sexual behaviour assumptions if one is fitting to HIV
	// prevalence data, after having fitted to sexual behaviour data
	if(SexHIVcount>0 && HIVcalib==1){
		HighPropnM = LSforSex[0];
		HighPropnF = LSforSex[1];
		PartnershipFormation[0][0] = LSforSex[8];
		PartnershipFormation[1][0] = 1.0;
		PartnershipFormation[0][1] = LSforSex[9];
		PartnershipFormation[1][1] = 1.0;
		for(ia=0; ia<16; ia++){
			AgeEffectPartners[ia][0] = LSforSex[2] * pow(LSforSex[4], LSforSex[6]) *
				pow(5.0 * ia + 2.5, LSforSex[6] - 1.0) * exp(-LSforSex[4] * 
				(5.0 * ia + 2.5));
			AgeEffectPartners[ia][1] = LSforSex[3] * pow(LSforSex[5], LSforSex[7]) *
				pow(5.0 * ia + 2.5, LSforSex[7] - 1.0) * exp(-LSforSex[5] * 
				(5.0 * ia + 2.5));
		}
		PartnerEffectNew[1][0] = LSforSex[10];
		PartnerEffectNew[1][1] = LSforSex[11];
	}
}

void ReadSTDepi(const char* input)
{
	int ia, is, iz;
	ifstream file;

	file.open(input);
	if(file==0)cout<<"File open error"<<endl;
	file.ignore(255,'\n');
	file>>HSVtransitionM.AveDuration[0]>>HSVtransitionM.AveDuration[1]>>
		HSVtransitionM.AveDuration[2]>>HSVtransitionF.AveDuration[0]>>
		HSVtransitionF.AveDuration[1]>>HSVtransitionF.AveDuration[2];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=0; iz<6; iz++){
		file>>TPtransitionM.AveDuration[iz];}
	for(iz=0; iz<6; iz++){
		file>>TPtransitionF.AveDuration[iz];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HDtransitionM.AveDuration[0]>>HDtransitionM.AveDuration[1]>>
		HDtransitionF.AveDuration[0]>>HDtransitionF.AveDuration[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>NGtransitionM.AveDuration[0]>>NGtransitionM.AveDuration[1]>>
		NGtransitionF.AveDuration[0]>>NGtransitionF.AveDuration[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>CTtransitionM.AveDuration[0]>>CTtransitionM.AveDuration[1]>>
		CTtransitionF.AveDuration[0]>>CTtransitionF.AveDuration[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>TVtransitionM.AveDuration[0]>>TVtransitionM.AveDuration[1]>>
		TVtransitionF.AveDuration[0]>>TVtransitionF.AveDuration[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(is=0; is<5; is++){
		file>>HIVtransitionM.AveDuration[is];}
	for(is=0; is<5; is++){
		file>>HIVtransitionF.AveDuration[is];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>VCtransitionF.AveDuration[0]>>VCtransitionF.AveDuration[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=1; iz<4; iz++){
		file>>BVtransitionF.CtsTransition[iz][0];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>BVtransitionF.CtsTransition[0][1]>>BVtransitionF.CtsTransition[2][1]>>
		BVtransitionF.CtsTransition[3][1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HSVtransitionM.RecurrenceRate>>HSVtransitionF.RecurrenceRate>>
		VCtransitionF.RecurrenceRate;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HSVtransitionM.SymptomaticPropn>>HDtransitionM.SymptomaticPropn>>
		NGtransitionM.SymptomaticPropn>>CTtransitionM.SymptomaticPropn>>
		TVtransitionM.SymptomaticPropn>>HSVtransitionF.SymptomaticPropn>>
		HDtransitionF.SymptomaticPropn>>NGtransitionF.SymptomaticPropn>>
		CTtransitionF.SymptomaticPropn>>TVtransitionF.SymptomaticPropn>>
		BVtransitionF.SymptomaticPropn;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HSVtransitionM.TransmProb>>TPtransitionM.TransmProb>>HDtransitionM.TransmProb>>
		NGtransitionM.TransmProb>>CTtransitionM.TransmProb>>TVtransitionM.TransmProb>>
		HSVtransitionF.TransmProb>>TPtransitionF.TransmProb>>HDtransitionF.TransmProb>>
		NGtransitionF.TransmProb>>CTtransitionF.TransmProb>>TVtransitionF.TransmProb>>
		HSVtransitionM.TransmProbSW>>TPtransitionM.TransmProbSW>>HDtransitionM.TransmProbSW>>
		NGtransitionM.TransmProbSW>>CTtransitionM.TransmProbSW>>TVtransitionM.TransmProbSW;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(is=0; is<3; is++){
		file>>InitHIVtransm[is][0];}
	for(is=0; is<3; is++){
		file>>InitHIVtransm[is][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	// Note that in the next line we are reading the male parameters into the female
	// arrays and the female parameters into the male arrays. This is deliberate; it makes
	// things a lot simpler when calculating the InfectProb arrays from the TransProb arrays
	// (in the STDtransition class).
	file>>HSVtransitionF.CondomEff>>TPtransitionF.CondomEff>>HDtransitionF.CondomEff>>
		NGtransitionF.CondomEff>>CTtransitionF.CondomEff>>TVtransitionF.CondomEff>>
		HIVtransitionF.CondomEff>>HSVtransitionM.CondomEff>>TPtransitionM.CondomEff>>
		HDtransitionM.CondomEff>>NGtransitionM.CondomEff>>CTtransitionM.CondomEff>>
		TVtransitionM.CondomEff>>HIVtransitionM.CondomEff;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=0; iz<4; iz++){
		file>>HSVtransitionM.HIVinfecIncrease[iz];}
	for(iz=0; iz<4; iz++){
		file>>HSVtransitionF.HIVinfecIncrease[iz];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=0; iz<6; iz++){
		file>>TPtransitionM.HIVinfecIncrease[iz];}
	for(iz=0; iz<6; iz++){
		file>>TPtransitionF.HIVinfecIncrease[iz];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HDtransitionM.HIVinfecIncrease[0]>>HDtransitionM.HIVinfecIncrease[1]>>
		HDtransitionF.HIVinfecIncrease[0]>>HDtransitionF.HIVinfecIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>NGtransitionM.HIVinfecIncrease[0]>>NGtransitionM.HIVinfecIncrease[1]>>
		NGtransitionF.HIVinfecIncrease[0]>>NGtransitionF.HIVinfecIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>CTtransitionM.HIVinfecIncrease[0]>>CTtransitionM.HIVinfecIncrease[1]>>
		CTtransitionF.HIVinfecIncrease[0]>>CTtransitionF.HIVinfecIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>TVtransitionM.HIVinfecIncrease[0]>>TVtransitionM.HIVinfecIncrease[1]>>
		TVtransitionF.HIVinfecIncrease[0]>>TVtransitionF.HIVinfecIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(is=0; is<5; is++){
		file>>HIVtransitionM.HIVinfecIncrease[is];}
	for(is=0; is<5; is++){
		file>>HIVtransitionF.HIVinfecIncrease[is];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>VCtransitionF.HIVinfecIncrease[0]>>VCtransitionF.HIVinfecIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=0; iz<3; iz++){
		file>>BVtransitionF.HIVinfecIncrease[iz];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=0; iz<4; iz++){
		file>>HSVtransitionM.HIVsuscepIncrease[iz];}
	for(iz=0; iz<4; iz++){
		file>>HSVtransitionF.HIVsuscepIncrease[iz];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=0; iz<6; iz++){
		file>>TPtransitionM.HIVsuscepIncrease[iz];}
	for(iz=0; iz<6; iz++){
		file>>TPtransitionF.HIVsuscepIncrease[iz];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HDtransitionM.HIVsuscepIncrease[0]>>HDtransitionM.HIVsuscepIncrease[1]>>
		HDtransitionF.HIVsuscepIncrease[0]>>HDtransitionF.HIVsuscepIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>NGtransitionM.HIVsuscepIncrease[0]>>NGtransitionM.HIVsuscepIncrease[1]>>
		NGtransitionF.HIVsuscepIncrease[0]>>NGtransitionF.HIVsuscepIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>CTtransitionM.HIVsuscepIncrease[0]>>CTtransitionM.HIVsuscepIncrease[1]>>
		CTtransitionF.HIVsuscepIncrease[0]>>CTtransitionF.HIVsuscepIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>TVtransitionM.HIVsuscepIncrease[0]>>TVtransitionM.HIVsuscepIncrease[1]>>
		TVtransitionF.HIVsuscepIncrease[0]>>TVtransitionF.HIVsuscepIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>VCtransitionF.HIVsuscepIncrease[0]>>VCtransitionF.HIVsuscepIncrease[1];
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=0; iz<3; iz++){
		file>>BVtransitionF.HIVsuscepIncrease[iz];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	// Note that in the next few lines we are reading the male parameters into the female
	// arrays and the female parameters into the male arrays. This is deliberate; it makes
	// things a lot simpler when calculating the InfectProb arrays from the TransProb arrays
	// (in the STDtransition class).
	for(ia=0; ia<16; ia++){
		file>>HSVtransitionF.SuscepIncrease[ia]>>TPtransitionF.SuscepIncrease[ia]>>
			HDtransitionF.SuscepIncrease[ia]>>NGtransitionF.SuscepIncrease[ia]>>
			CTtransitionF.SuscepIncrease[ia]>>TVtransitionF.SuscepIncrease[ia]>>
			HIVtransitionF.SuscepIncrease[ia];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		file>>HSVtransitionM.SuscepIncrease[ia]>>TPtransitionM.SuscepIncrease[ia]>>
			HDtransitionM.SuscepIncrease[ia]>>NGtransitionM.SuscepIncrease[ia]>>
			CTtransitionM.SuscepIncrease[ia]>>TVtransitionM.SuscepIncrease[ia]>>
			HIVtransitionM.SuscepIncrease[ia];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(is=0; is<5; is++){
		file>>HSVsheddingIncrease[is];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(is=0; is<5; is++){
		file>>HSVrecurrenceIncrease[is];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(is=0; is<5; is++){
		file>>VCtransitionF.IncidenceIncrease[is];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>VCtransitionF.Incidence>>BVtransitionF.Incidence1;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>BVtransitionF.IncidenceMultTwoPartners>>BVtransitionF.IncidenceMultNoPartners;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HSVsymptomInfecIncrease;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=0; iz<3; iz++){
		file>>InfecIncreaseSyndrome[iz][0];}
	for(iz=0; iz<3; iz++){
		file>>InfecIncreaseSyndrome[iz][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iz=0; iz<3; iz++){
		file>>SuscepIncreaseSyndrome[iz][0];}
	for(iz=0; iz<3; iz++){
		file>>SuscepIncreaseSyndrome[iz][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(is=0; is<5; is++){
		file>>RelHIVfertility[is];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>PropnInfectedAtBirth>>PropnInfectedAfterBirth;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>MaleChild.Perinatal.PreAIDSmedian>>FemChild.Perinatal.PreAIDSmedian>>
		MaleChild.Breastmilk.PreAIDSmedian>>FemChild.Breastmilk.PreAIDSmedian;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>MaleChild.Perinatal.PreAIDSshape>>FemChild.Perinatal.PreAIDSshape>>
		MaleChild.Breastmilk.PreAIDSshape>>FemChild.Breastmilk.PreAIDSshape;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>MaleChild.Perinatal.MeanAIDSsurvival>>FemChild.Perinatal.MeanAIDSsurvival>>
		MaleChild.Breastmilk.MeanAIDSsurvival>>FemChild.Breastmilk.MeanAIDSsurvival;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>MaleTeenRxRate>>MaleRxRate>>FemTeenRxRate>>FemRxRate>>FSWRxRate;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>PropnTreatedPublicM>>PropnTreatedPublicF>>PropnTreatedPrivateM>>
		PropnTreatedPrivateF;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HSVtransitionF.CorrectRxPreSM>>TPtransitionF.CorrectRxPreSM>>
		HDtransitionF.CorrectRxPreSM>>NGtransitionF.CorrectRxPreSM>>
		CTtransitionF.CorrectRxPreSM>>TVtransitionF.CorrectRxPreSM>>
		BVtransitionF.CorrectRxPreSM>>VCtransitionF.CorrectRxPreSM;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HSVtransitionM.CorrectRxWithSM>>TPtransitionM.CorrectRxWithSM>>
		HDtransitionM.CorrectRxWithSM>>NGtransitionM.CorrectRxWithSM>>
		CTtransitionM.CorrectRxWithSM>>TVtransitionM.CorrectRxWithSM>>
		HSVtransitionF.CorrectRxWithSM>>TPtransitionF.CorrectRxWithSM>>
		HDtransitionF.CorrectRxWithSM>>NGtransitionF.CorrectRxWithSM>>
		CTtransitionF.CorrectRxWithSM>>TVtransitionF.CorrectRxWithSM>>
		BVtransitionF.CorrectRxWithSM;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HSVtransitionF.DrugEff>>TPtransitionF.DrugEff>>HDtransitionF.DrugEff>>
		NGtransitionF.DrugEff>>CTtransitionF.DrugEff>>TVtransitionF.DrugEff>>
		BVtransitionF.DrugEff>>VCtransitionF.DrugEff;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>BVtransitionF.DrugPartialEff>>VCtransitionF.DrugPartialEff;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>HSVtransitionF.TradnalEff>>TPtransitionF.TradnalEff>>HDtransitionF.TradnalEff>>
		NGtransitionF.TradnalEff>>CTtransitionF.TradnalEff>>TVtransitionF.TradnalEff>>
		BVtransitionF.TradnalEff>>VCtransitionF.TradnalEff;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>TPtransitionF.ANCpropnScreened>>TPtransitionF.ANCpropnTreated;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>AcceptScreening>>AcceptNVP>>RednNVP>>RednFF;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>SecondaryRxMult>>SecondaryCureMult;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>FSWasympRxRate>>FSWasympCure;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>InitHIVprevHigh;
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>RatioUltToInitHIVtransm;
	file.close();

	// Set the effectiveness of SM for VC
	VCtransitionF.CorrectRxWithSM = VCtransitionF.CorrectRxPreSM;

	// Specify parameter values that apply to both sexes.
	HSVtransitionM.CorrectRxPreSM = HSVtransitionF.CorrectRxPreSM;
	TPtransitionM.CorrectRxPreSM = TPtransitionF.CorrectRxPreSM;
	HDtransitionM.CorrectRxPreSM = HDtransitionF.CorrectRxPreSM;
	NGtransitionM.CorrectRxPreSM = NGtransitionF.CorrectRxPreSM;
	CTtransitionM.CorrectRxPreSM = CTtransitionF.CorrectRxPreSM;
	TVtransitionM.CorrectRxPreSM = TVtransitionF.CorrectRxPreSM;

	HSVtransitionM.DrugEff = HSVtransitionF.DrugEff;
	TPtransitionM.DrugEff = TPtransitionF.DrugEff;
	HDtransitionM.DrugEff = HDtransitionF.DrugEff;
	NGtransitionM.DrugEff = NGtransitionF.DrugEff;
	CTtransitionM.DrugEff = CTtransitionF.DrugEff;
	TVtransitionM.DrugEff = TVtransitionF.DrugEff;

	HSVtransitionM.TradnalEff = HSVtransitionF.TradnalEff;
	TPtransitionM.TradnalEff = TPtransitionF.TradnalEff;
	HDtransitionM.TradnalEff = HDtransitionF.TradnalEff;
	NGtransitionM.TradnalEff = NGtransitionF.TradnalEff;
	CTtransitionM.TradnalEff = CTtransitionF.TradnalEff;
	TVtransitionM.TradnalEff = TVtransitionF.TradnalEff;

	// Convert annualized recurrence and incidence rates into weekly rates
	HSVtransitionM.RecurrenceRate = HSVtransitionM.RecurrenceRate/52.0;
	HSVtransitionF.RecurrenceRate = HSVtransitionF.RecurrenceRate/52.0;
	VCtransitionF.RecurrenceRate = VCtransitionF.RecurrenceRate/52.0;
	VCtransitionF.Incidence = VCtransitionF.Incidence/52.0;

	// Calculate remaining elements of the CtsTransition matrix and AveDuration for BV
	BVtransitionF.CtsTransition[1][2] = BVtransitionF.Incidence1 * 
		BVtransitionF.SymptomaticPropn;
	BVtransitionF.CtsTransition[1][3] = BVtransitionF.Incidence1 * 
		(1.0 - BVtransitionF.SymptomaticPropn);
	BVtransitionF.AveDuration[0] = 1.0/BVtransitionF.CtsTransition[0][1];
	BVtransitionF.AveDuration[1] = 1.0/(BVtransitionF.CtsTransition[1][0] + 
		BVtransitionF.CtsTransition[1][2] + BVtransitionF.CtsTransition[1][3]);
	BVtransitionF.AveDuration[2] = 1.0/(BVtransitionF.CtsTransition[2][0] + 
		BVtransitionF.CtsTransition[2][1]);
	BVtransitionF.AveDuration[3] = 1.0/(BVtransitionF.CtsTransition[3][0] + 
		BVtransitionF.CtsTransition[3][1]);

	// Calculate RatioAsympToAveM and RatioAsympToAveF
	double sumx, sumxy;
	sumx = 0;
	sumxy = 0;
	for(is=0; is<4; is++){
		sumx += HIVtransitionM.AveDuration[is];
		sumxy += HIVtransitionM.AveDuration[is] * HIVtransitionM.HIVinfecIncrease[is];
	}
	RatioAsympToAveM = sumx /(sumx + sumxy);
	sumx = 0;
	sumxy = 0;
	for(is=0; is<4; is++){
		sumx += HIVtransitionF.AveDuration[is];
		sumxy += HIVtransitionF.AveDuration[is] * HIVtransitionM.HIVinfecIncrease[is];
	}
	RatioAsympToAveF = sumx /(sumx + sumxy);

	// Calculate rates of progression to AIDS in children
	MaleChild.Perinatal.CalcAIDSprogression();
	MaleChild.Breastmilk.CalcAIDSprogression();
	FemChild.Perinatal.CalcAIDSprogression();
	FemChild.Breastmilk.CalcAIDSprogression();

	// Set the HIV transmission probs in current year to their initial values
	HIVtransitionM.TransmProb[0] = InitHIVtransm[0][0];
	HIVtransitionF.TransmProb[0] = InitHIVtransm[0][1];
	for(is=0; is<3; is++){
		HIVtransitionM.TransmProb[is+1] = InitHIVtransm[1][0];
		HIVtransitionM.TransmProb[is+4] = InitHIVtransm[2][0];
		HIVtransitionF.TransmProb[is+1] = InitHIVtransm[1][1];
		HIVtransitionF.TransmProb[is+4] = InitHIVtransm[2][1];
	}
	if(CofactorType==0){
		HIVtransitionM.TransmProb[1] *= 2.0;
		HIVtransitionM.TransmProb[3] *= 0.5;
		HIVtransitionM.TransmProb[4] *= 2.0;
		HIVtransitionM.TransmProb[6] *= 0.5;
		HIVtransitionF.TransmProb[1] *= 2.0;
		HIVtransitionF.TransmProb[3] *= 0.5;
		HIVtransitionF.TransmProb[4] *= 2.0;
		HIVtransitionF.TransmProb[6] *= 0.5;
	}

	// Set the RelTransmCSW values
	HSVtransitionM.RelTransmCSW = HSVtransitionM.TransmProbSW/HSVtransitionM.TransmProb;
	TPtransitionM.RelTransmCSW = TPtransitionM.TransmProbSW/TPtransitionM.TransmProb;
	HDtransitionM.RelTransmCSW = HDtransitionM.TransmProbSW/HDtransitionM.TransmProb;
	NGtransitionM.RelTransmCSW = NGtransitionM.TransmProbSW/NGtransitionM.TransmProb;
	CTtransitionM.RelTransmCSW = CTtransitionM.TransmProbSW/CTtransitionM.TransmProb;
	TVtransitionM.RelTransmCSW = TVtransitionM.TransmProbSW/TVtransitionM.TransmProb;
}

void ReadRatesByYear()
{
	int iy;
	ifstream file;

	file.open("RatesByYear.txt");
	if(file==0)cout<<"File open error"<<endl;
	file.ignore(255,'\n');
	for(iy=0; iy<41; iy++){
		file>>PropnPrivateUsingSM[iy];}
	for(iy=0; iy<41; iy++){
		file>>PropnPublicUsingSM[iy];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iy=0; iy<41; iy++){
		file>>DrugShortage[iy];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iy=0; iy<41; iy++){
		file>>HAARTaccess[iy];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iy=0; iy<41; iy++){
		file>>PMTCTaccess[iy];}
	file.close();
}

void ReadMortTables()
{
	int ia, iy;
	ifstream file;

	file.open("MortTables.txt");
	if(file==0)cout<<"File open error"<<endl;
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		for(iy=0; iy<41; iy++){
			file>>NonAIDSmortM[ia][iy];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		for(iy=0; iy<41; iy++){
			file>>NonAIDSmortF[ia][iy];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iy=0; iy<41; iy++){
		file>>InfantMort1st6mM[iy];}
	for(ia=0; ia<15; ia++){
		for(iy=0; iy<41; iy++){
			file>>ChildMortM[ia][iy];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(iy=0; iy<41; iy++){
		file>>InfantMort1st6mF[iy];}
	for(ia=0; ia<15; ia++){
		for(iy=0; iy<41; iy++){
			file>>ChildMortF[ia][iy];}
	}
	file.close();
}

void ReadFertTables()
{
	int ia, iy;
	ifstream file;

	file.open("FertTables.txt");
	if(file==0)cout<<"File open error"<<endl;
	file.ignore(255,'\n');
	for(ia=0; ia<7; ia++){
		for(iy=0; iy<41; iy++){
			file>>FertilityTable[ia][iy];}
	}
	file.close();
}

void ReadAgeExitRates()
{
	int ia, is;
	ifstream file;

	file.open("AgeExitRates.txt");
	if(file==0)cout<<"File open error"<<endl;
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		for(is=0; is<6; is++){
			file>>AgeExitRateM[ia][is];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		for(is=0; is<6; is++){
			file>>AgeExitRateF[ia][is];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<16; ia++){
		file>>VirginAgeExitRate[ia][0]>>VirginAgeExitRate[ia][1];}
	file.close();
}

void ReadOneStartProfileM(ifstream* file, SexuallyExpM* m)
{
	int ia, iz;

	for(ia=0; ia<16; ia++){
		*file>>m->TotalAlive[ia];
		for(iz=0; iz<5; iz++){
			*file>>m->MHSV.PropnByStage0[ia][iz];}
		for(iz=0; iz<7; iz++){
			*file>>m->MTP.PropnByStage0[ia][iz];}
		for(iz=0; iz<3; iz++){
			*file>>m->MHD.PropnByStage0[ia][iz];}
		for(iz=0; iz<3; iz++){
			*file>>m->MNG.PropnByStage0[ia][iz];}
		for(iz=0; iz<3; iz++){
			*file>>m->MCT.PropnByStage0[ia][iz];}
		for(iz=0; iz<3; iz++){
			*file>>m->MTV.PropnByStage0[ia][iz];}
	}
}

void ReadOneStartProfileF(ifstream* file, SexuallyExpF* f)
{
	int ia, iz;

	for(ia=0; ia<16; ia++){
		*file>>f->TotalAlive[ia];
		for(iz=0; iz<5; iz++){
			*file>>f->FHSV.PropnByStage0[ia][iz];}
		for(iz=0; iz<7; iz++){
			*file>>f->FTP.PropnByStage0[ia][iz];}
		for(iz=0; iz<3; iz++){
			*file>>f->FHD.PropnByStage0[ia][iz];}
		for(iz=0; iz<3; iz++){
			*file>>f->FNG.PropnByStage0[ia][iz];}
		for(iz=0; iz<3; iz++){
			*file>>f->FCT.PropnByStage0[ia][iz];}
		for(iz=0; iz<3; iz++){
			*file>>f->FTV.PropnByStage0[ia][iz];}
		for(iz=0; iz<3; iz++){
			*file>>f->FVC.PropnByStage0[ia][iz];}
		for(iz=0; iz<4; iz++){
			*file>>f->FBV.PropnByStage0[ia][iz];}
	}
}

void ReadStartProfile(const char* input)
{
	int ia, iz;
	ifstream file;

	file.open(input);
	if(file==0)cout<<"File open error"<<endl;
	file.ignore(255,'\n');
	for(ia=0; ia<4; ia++){
			file>>MaleHigh.Virgin.TotalAlive[ia];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<4; ia++){
			file>>MaleLow.Virgin.TotalAlive[ia];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<4; ia++){
		file>>FemHigh.Virgin.TotalAlive[ia];
		for(iz=0; iz<3; iz++){
			file>>FemHigh.Virgin.FVC.PropnByStage0[ia][iz];}
		for(iz=0; iz<4; iz++){
			file>>FemHigh.Virgin.FBV.PropnByStage0[ia][iz];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<4; ia++){
		file>>FemLow.Virgin.TotalAlive[ia];
		for(iz=0; iz<3; iz++){
			file>>FemLow.Virgin.FVC.PropnByStage0[ia][iz];}
		for(iz=0; iz<4; iz++){
			file>>FemLow.Virgin.FBV.PropnByStage0[ia][iz];}
	}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.NoPartner);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.S1);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.S2);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.L1);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.L2);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.S11);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.S12);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.S22);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.L11);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.L12);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.L21);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleHigh.L22);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleLow.NoPartner);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleLow.S1);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleLow.S2);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleLow.L1);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileM(&file, &MaleLow.L2);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.FSW);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.NoPartner);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.S1);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.S2);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.L1);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.L2);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.S11);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.S12);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.S22);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.L11);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.L12);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.L21);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemHigh.L22);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemLow.NoPartner);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemLow.S1);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemLow.S2);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemLow.L1);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	ReadOneStartProfileF(&file, &FemLow.L2);
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<10; ia++){
		file>>MaleChild.HIVneg[ia]>>FemChild.HIVneg[ia];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	file>>MaleChild.PropnBirths;
	file.close();

	// Set the proportions of uninfected virgins to 100%.
	for(ia=0; ia<4; ia++){
		if(HSVind==1){
			MaleHigh.Virgin.MHSV.PropnByStage0[ia][0] = 1.0;
			MaleLow.Virgin.MHSV.PropnByStage0[ia][0] = 1.0;
			FemHigh.Virgin.FHSV.PropnByStage0[ia][0] = 1.0;
			FemLow.Virgin.FHSV.PropnByStage0[ia][0] = 1.0;
		}
		if(TPind==1){
			MaleHigh.Virgin.MTP.PropnByStage0[ia][0] = 1.0;
			MaleLow.Virgin.MTP.PropnByStage0[ia][0] = 1.0;
			FemHigh.Virgin.FTP.PropnByStage0[ia][0] = 1.0;
			FemLow.Virgin.FTP.PropnByStage0[ia][0] = 1.0;
		}
		if(HDind==1){
			MaleHigh.Virgin.MHD.PropnByStage0[ia][0] = 1.0;
			MaleLow.Virgin.MHD.PropnByStage0[ia][0] = 1.0;
			FemHigh.Virgin.FHD.PropnByStage0[ia][0] = 1.0;
			FemLow.Virgin.FHD.PropnByStage0[ia][0] = 1.0;
		}
		if(NGind==1){
			MaleHigh.Virgin.MNG.PropnByStage0[ia][0] = 1.0;
			MaleLow.Virgin.MNG.PropnByStage0[ia][0] = 1.0;
			FemHigh.Virgin.FNG.PropnByStage0[ia][0] = 1.0;
			FemLow.Virgin.FNG.PropnByStage0[ia][0] = 1.0;
		}
		if(CTind==1){
			MaleHigh.Virgin.MCT.PropnByStage0[ia][0] = 1.0;
			MaleLow.Virgin.MCT.PropnByStage0[ia][0] = 1.0;
			FemHigh.Virgin.FCT.PropnByStage0[ia][0] = 1.0;
			FemLow.Virgin.FCT.PropnByStage0[ia][0] = 1.0;
		}
		if(TVind==1){
			MaleHigh.Virgin.MTV.PropnByStage0[ia][0] = 1.0;
			MaleLow.Virgin.MTV.PropnByStage0[ia][0] = 1.0;
			FemHigh.Virgin.FTV.PropnByStage0[ia][0] = 1.0;
			FemLow.Virgin.FTV.PropnByStage0[ia][0] = 1.0;
		}
	}

	// Set the proportion of births that are female = 1 - male proportion
	FemChild.PropnBirths = 1.0 - MaleChild.PropnBirths;
}

void ReadStartPop()
{
	int ia;
	ifstream file;

	file.open("StartPop.txt");
	if(file==0)cout<<"File open error"<<endl;
	for(ia=0; ia<81; ia++){
		file>>StartPop[ia][0]>>StartPop[ia][1];}
	file.close();
}

void ReadSTDprev()
{
	if(HIVcalib==1){
		HIVtransitionM.ReadPrevData("HIVdataM.txt");
		HIVtransitionF.ReadPrevData("HIVdataF.txt");
	}
	if(HSVcalib==1){
		HSVtransitionM.ReadPrevData("HSVdataM.txt");
		HSVtransitionF.ReadPrevData("HSVdataF.txt");
	}
	if(TPcalib==1){
		TPtransitionM.ReadPrevData("TPdataM.txt");
		TPtransitionF.ReadPrevData("TPdataF.txt");
	}
	if(HDcalib==1){
		HDtransitionM.ReadPrevData("HDdataM.txt");
		HDtransitionF.ReadPrevData("HDdataF.txt");
	}
	if(NGcalib==1){
		NGtransitionM.ReadPrevData("NGdataM.txt");
		NGtransitionF.ReadPrevData("NGdataF.txt");
	}
	if(CTcalib==1){
		CTtransitionM.ReadPrevData("CTdataM.txt");
		CTtransitionF.ReadPrevData("CTdataF.txt");
	}
	if(TVcalib==1){
		TVtransitionM.ReadPrevData("TVdataM.txt");
		TVtransitionF.ReadPrevData("TVdataF.txt");
	}
	if(BVcalib==1){BVtransitionF.ReadPrevData("BVdataF.txt");}
	if(VCcalib==1){VCtransitionF.ReadPrevData("VCdataF.txt");}
}

void ReadSexData(const char* input)
{
	int ia;
	ifstream file;

	file.open(input);
	if(file==0)cout<<"File open error"<<endl;
	file.ignore(255,'\n');
	for(ia=0; ia<2; ia++){
		file>>VirginPropnC[ia][0]>>VirginPropnC[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<2; ia++){
		file>>VirginPropnSD[ia][0]>>VirginPropnSD[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<5; ia++){
		file>>UnmarriedMultPartnersC[ia][0]>>UnmarriedMultPartnersC[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<5; ia++){
		file>>UnmarriedMultPartnersSD[ia][0]>>UnmarriedMultPartnersSD[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<5; ia++){
		file>>MarriedMultPartnersC[ia][0]>>MarriedMultPartnersC[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<5; ia++){
		file>>MarriedMultPartnersSD[ia][0]>>MarriedMultPartnersSD[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<5; ia++){
		file>>UnmarriedSingleC[ia][0]>>UnmarriedSingleC[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<5; ia++){
		file>>UnmarriedSingleSD[ia][0]>>UnmarriedSingleSD[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<5; ia++){
		file>>MarriedPropn95C[ia][0]>>MarriedPropn95C[ia][1];}
	file.ignore(255,'\n');
	file.ignore(255,'\n');
	for(ia=0; ia<7; ia++){
		file>>MarriedPropn98C[ia];}
	file.close();
}

void ReadAllInputFiles()
{
	ReadSexAssumps("SexAssumps.txt");
	ReadSTDepi("STDepidemiology.txt");
	ReadRatesByYear();
	ReadMortTables();
	ReadFertTables();
	ReadAgeExitRates();
	ReadStartProfile("StartProfile.txt");
	ReadStartPop();
	ReadSTDprev();
	if(SexCalib==1){ReadSexData("SexData.txt");}
}

void ReadRandomAdj()
{
	int ic, ir;
	ifstream file1, file2;

	file1.open("RandomAdjHIV.txt");
	for(ir=0; ir<8; ir++){
		for(ic=0; ic<7; ic++){
			file1>>RandomAdjHIV[ir][ic];}
	}
	file1.close();

	file2.open("RandomAdjSex.txt");
	for(ir=0; ir<13; ir++){
		for(ic=0; ic<12; ic++){
			file2>>RandomAdjSex[ir][ic];}
	}
	file2.close();
}

double GetDebutRate(int ia, int sex, double start, double tolerance)
{
	// In this function we are using Newton's method to determine numerically the rate of
	// sexual debut in the high risk group (DebutRate) that applies over the age interval
	// (x + 0.5, x + 5.5), using the function f(DebutRate) = Observed virgin propn in 
	// (x, x + 5) interval - modelled virgin propn in (x, x + 5) interval. The observed
	// virgin propn is referred to as ObsVPropn in the code below (but note that it is
	// actually adjusted for social desirability mis-reporting). It is assumed that we
	// already know the rates of sexual debut in the previous age bands, which are used to
	// determine the modelled proportions of individuals who are virgins at age x + 0.5 in
	// the high and low risk groups (VirginPropnStart[2]).

	double HighPropn, ObsVPropn;
	double VirginPropnStart[2]; 
	double DebutRate, NewDebutRate, LowDebutRate;
	double fDebutRate, fPrimeDebutRate;

	// Set the initial parameter values
	if(sex==0){HighPropn = HighPropnM;}
	else{HighPropn = HighPropnF;}
	ObsVPropn = (VirginPropnC[ia-1][sex]/DebutBias[ia-1][sex])/(VirginPropnC[ia-1][sex] *
		(1.0/DebutBias[ia-1][sex] - 1.0) + 1.0);
	if(ia==1){
		VirginPropnStart[0] = exp(-5.0 * SexualDebut[0][sex]);}
	if(ia==2){
		VirginPropnStart[0] = exp(-5.0 * (SexualDebut[0][sex] + SexualDebut[1][sex]));}
	VirginPropnStart[1] = pow(VirginPropnStart[0], DebutAdjLow[sex]);
	DebutRate = 0.1;
	NewDebutRate = start;

	// Iterate through the steps of Newton's method
	while(fabs(NewDebutRate - DebutRate) > tolerance){
		DebutRate = NewDebutRate;
		LowDebutRate = NewDebutRate * DebutAdjLow[sex];
		fDebutRate = ObsVPropn - 0.2 * (HighPropn * VirginPropnStart[0] * (0.5 + (1.0 -
			exp(-4.5 * DebutRate))/DebutRate) + (1.0 - HighPropn) * VirginPropnStart[1] *
			(0.5 + (1.0 - exp(-4.5 * LowDebutRate))/LowDebutRate));
		fPrimeDebutRate = 0.2 * (HighPropn * VirginPropnStart[0] * ((1.0 - exp(-4.5 *
			DebutRate))/pow(DebutRate, 2.0) - 4.5 * exp(-4.5 * DebutRate)/DebutRate) + 
			(1.0 - HighPropn) * VirginPropnStart[1] * ((1.0 - exp(-4.5 * LowDebutRate))/
			(DebutAdjLow[sex] * pow(DebutRate, 2.0)) - 4.5 * exp(-4.5 * LowDebutRate)/
			DebutRate));
		NewDebutRate = DebutRate - fDebutRate/fPrimeDebutRate;
	}

	// Calculate the SexuallyExpPropn values for the GetPartnerAcqHigh() function
	LowDebutRate = NewDebutRate * DebutAdjLow[sex];
	SexuallyExpPropnH[ia-1][sex] = 1.0 - VirginPropnStart[0] * (0.5 + (1.0 - exp(-4.5 * 
		NewDebutRate))/NewDebutRate) * 0.2;
	SexuallyExpPropnL[ia-1][sex] = 1.0 - VirginPropnStart[1] * (0.5 + (1.0 - exp(-4.5 * 
		LowDebutRate))/LowDebutRate) * 0.2;

	return NewDebutRate;
}

double GetPartnerAcqHigh(int sex, double start, double tolerance)
{
	// In this function we are using Newton's method to determine numerically the rate of
	// partner acquisition in the high risk group (c) that applies over the 15-24 age
	// range, using the function f(c) = Observed propn of sexually experienced with >1 
	// partner - modelled propn. The observed propn with >1 partner is referred to as 
	// ObsMultPropn in the code below (but note that it is actually adjusted for social 
	// desirability mis-reporting).

	double ObsMultPropn;
	double c, newc;
	double fc, fPrimec;

	ObsMultPropn = UnmarriedMultPartnersC[0][sex] * ConcurrencyBias[0][sex]/
		(UnmarriedMultPartnersC[0][sex] * (ConcurrencyBias[0][sex] - 1.0) + 1.0);
	if(sex==0 && ObsMultPropn > HighPropnM){
		ErrorInd = 1;}
	if(sex==1 && ObsMultPropn > HighPropnF){
		ErrorInd = 1;}
	c = 2.0;
	newc = start;

	// Iterate through the steps of Newton's method
	if(ErrorInd==0){
		while(fabs(newc - c) > tolerance){
			c = newc;
			fc = ObsMultPropn - HighPropn15to24[sex] * 0.5 * PartnerEffectNew[0][sex] *
				pow(c * MeanDurSTrel[0][0], 2.0)/(1.0 + c * MeanDurSTrel[0][0] + 0.5 *
				PartnerEffectNew[0][sex] * pow(c * MeanDurSTrel[0][0], 2.0));
			fPrimec = -HighPropn15to24[sex] * PartnerEffectNew[0][sex] * c * 
				pow(MeanDurSTrel[0][0], 2.0) * (1.0 + 0.5 * c * MeanDurSTrel[0][0])/pow(1.0 +
				c * MeanDurSTrel[0][0] + 0.5 * PartnerEffectNew[0][sex] * pow(c * 
				MeanDurSTrel[0][0], 2.0), 2.0);
			if(fPrimec==0.0){
				ErrorInd = 1;
				break;}
			else{
				newc = c - fc/fPrimec;}
		}
	}
	if(newc<0.0){
		ErrorInd = 1;}
	if(ErrorInd==1){
		newc = start;}

	return newc;
}

void GetAllPartnerRates()
{
	// In addition to calculating the rates of partnership formation by age, sex and risk
	// group, this function checks that the consistency constraints have been met, and sets
	// the ErrorInd indicator to 1 if they haven't. There are 4 sets of constraints:
	// (i) Rate of partner acquisition in single high risk females aged 15-19 must be > that
	//	   in single males of the same age and risk group
	// (ii) Rate of partner acquisition in married high risk individuals must be < that in
	//		unmarried individuals in a non-spousal relationship
	// (iii) Reduction in partner acquisition after marriage, among high risk females, must
	//		 be > that in high risk males
	// (iv) The high risk proportion in males must be > that in females

	int ia, ig;
	double lambda, alpha;
	double GammaScaling[2];

	// Get the rates of sexual debut
	/*for(ig=0; ig<2; ig++){
		for(ia=1; ia<3; ia++){
			SexualDebut[ia][ig] = GetDebutRate(ia, ig, 0.5, 0.0001);
			cout<<"SexualDebut["<<ia<<"]["<<ig<<"]: "<<SexualDebut[ia][ig]<<endl;
		}
	}*/

	// Calculate the age adjustment factors and the high risk partner acquisition rates
	for(ig=0; ig<2; ig++){
		lambda = (GammaMeanST[ig] - 10.0)/pow(GammaStdDevST[ig], 2.0);
		alpha = (GammaMeanST[ig] - 10.0) * lambda;
		for(ia=0; ia<16; ia++){
			AgeEffectPartners[ia][ig] = pow(lambda, alpha) * pow(ia*5.0 + 2.5, alpha - 1.0) *
				exp(-lambda * (ia*5.0 + 2.5));
		}
		GammaScaling[ig] = BasePartnerAcqH[ig]/AgeEffectPartners[1][ig];
		for(ia=0; ia<16; ia++){
			AgeEffectPartners[ia][ig] *= GammaScaling[ig];}
		PartnershipFormation[0][ig] = 1.0;
	}
	if(BasePartnerAcqH[0] > BasePartnerAcqH[1]){ErrorInd = 1;}

	// Check the reduction in partner acquisition after marriage
	for(ig=0; ig<2; ig++){
		if(PartnerEffectNew[1][ig] > PartnerEffectNew[0][ig]){ErrorInd = 1;}
	}
	if(PartnerEffectNew[1][1] > PartnerEffectNew[1][0]){ErrorInd = 1;}

	// Also check the male high risk proportion > female high risk proportion
	if(HighPropnF > HighPropnM){ErrorInd = 1;}

	if(ErrorInd==1){ErrorCount += 1;}
}

void GetStartProfile()
{
	// This function performs the same calculations as in the "Initial profile" sheet of
	// SexBehavWorkings4.xls (in the Sex scenarios folder), in order to generate the number
	// of individuals in each sexual behaviour class at the start of the projection.

	int ia, ib, ig, iy, ii, ij, count;
	double SpouseMortM, SpouseMortF;
	double HighPropn;
	double VirginDenom, SexuallyExpDenom;
	double cHD, cLD; // Rates of partnership formation x ave duration in high & low risk
	double ActualPropnSTH[2][2]; // Actual propns of ST partners in high risk group 
								 // (1st index: risk group; 2nd index: sex)
	double alpha, lambda; // Parameters for the gamma dbn used to determine relative
						  // frequencies of FSW contact at different ages
	double FSWcontactDemand[16], FSWcontactBase[16];
	double BaseFSWdemand, AdjFSWdemand, MalePop15to49;

	// In the arrays defined below, the I- prefix indicates individual ages and the G-
	// prefix indicates data grouped in quinquennial age bands. The -H suffix indicates high
	// risk and the -L suffix indicates low risk. The -0, -1 and -2 suffixes indicate the
	// current number of partners.
	double IMarriedPropn[81][2], IVirginPropnH[81][2], IVirginPropnL[81][2];
	double IVirginsH[81][2], IVirginsL[81][2];
	double ISexuallyExpH[81][2], ISexuallyExpL[81][2];
	double IMarriedH[81][2], IMarriedL[81][2];
	double ISexuallyExpUnmarriedH[81][2], ISexuallyExpUnmarriedL[81][2];
	double GVirginsH[16][2], GVirginsL[16][2];
	double GSexuallyExpH[16][2], GSexuallyExpL[16][2];
	double GMarriedH[16][2], GMarriedL[16][2];
	double GSexuallyExpUnmarriedH[16][2], GSexuallyExpUnmarriedL[16][2];
	double GUnmarriedH0[16][2], GUnmarriedH1[16][2], GUnmarriedH2[16][2]; 
	double GUnmarriedL0[16][2], GUnmarriedL1[16][2];
	double GMarriedH1[16][2], GMarriedH2[16][2];

	// First calculate IMarriedPropn and IVirginPropn
	IMarriedPropn[0][0] = 0.0;
	IMarriedPropn[0][1] = 0.0;
	IVirginPropnH[0][0] = 1.0;
	IVirginPropnH[0][1] = 1.0;
	IVirginPropnL[0][0] = 1.0;
	IVirginPropnL[0][1] = 1.0;
	for(ia=1; ia<81; ia++){
		SpouseMortM = 0;
		SpouseMortF = 0;
		for(ib=0; ib<16; ib++){
			SpouseMortM += AgePrefM[(ia-1)/5][ib] * NonAIDSmortF[ib][0];
			SpouseMortF += AgePrefF[(ia-1)/5][ib] * NonAIDSmortM[ib][0];
		}
		IMarriedPropn[ia][0] = (1.0 - IMarriedPropn[ia-1][0]) * (1.0 - exp(
			-MarriageIncidence[(ia-1)/5][0])) + IMarriedPropn[ia-1][0] * exp(
			-LTseparation[(ia-1)/5][0]) * (1.0 - SpouseMortM);
		IMarriedPropn[ia][1] = (1.0 - IMarriedPropn[ia-1][1]) * (1.0 - exp(
			-MarriageIncidence[(ia-1)/5][1])) + IMarriedPropn[ia-1][1] * exp(
			-LTseparation[(ia-1)/5][1]) * (1.0 - SpouseMortF);
		for(ig=0; ig<2; ig++){
			if(ia<20){
				IVirginPropnH[ia][ig] = IVirginPropnH[ia-1][ig] * 
					exp(-SexualDebut[(ia-1)/5][ig]);
				IVirginPropnL[ia][ig] = IVirginPropnL[ia-1][ig] * 
					exp(-SexualDebut[(ia-1)/5][ig] * DebutAdjLow[ig]);
			}
			else{
				IVirginPropnH[ia][ig] = 0;
				IVirginPropnL[ia][ig] = 0;
			}
		}
	}

	// Next calculate the individual-age statistics
	for(ig=0; ig<2; ig++){
		if(ig==0){HighPropn = HighPropnM;}
		else{HighPropn = HighPropnF;}
		for(ia=0; ia<81; ia++){
			IVirginsH[ia][ig] = StartPop[ia][ig] * HighPropn * IVirginPropnH[ia][ig];
			IVirginsL[ia][ig] = StartPop[ia][ig] * (1.0 - HighPropn) * IVirginPropnL[ia][ig];
			ISexuallyExpH[ia][ig] = StartPop[ia][ig] * HighPropn - IVirginsH[ia][ig];
			ISexuallyExpL[ia][ig] = StartPop[ia][ig] * (1.0 - HighPropn) - IVirginsL[ia][ig];
			IMarriedH[ia][ig] = StartPop[ia][ig] * HighPropn * IMarriedPropn[ia][ig];
			IMarriedL[ia][ig] = StartPop[ia][ig] * (1.0 - HighPropn) * IMarriedPropn[ia][ig];
			ISexuallyExpUnmarriedH[ia][ig] = ISexuallyExpH[ia][ig] - IMarriedH[ia][ig];
			ISexuallyExpUnmarriedL[ia][ig] = ISexuallyExpL[ia][ig] - IMarriedL[ia][ig];
		}
	}

	// Revise the age exit rates for the age bands <25, taking into account the revised
	// estimates of the propns of youth who are virgins at each age.
	for(ia=0; ia<3; ia++){
		for(ig=0; ig<2; ig++){
			VirginDenom = IVirginsH[ia*5+1][ig] + IVirginsL[ia*5+1][ig];
			SexuallyExpDenom = ISexuallyExpH[ia*5+1][ig] + ISexuallyExpL[ia*5+1][ig];
			for(iy=2; iy<=5; iy++){
				VirginDenom += IVirginsH[ia*5+iy][ig] + IVirginsL[ia*5+iy][ig];
				SexuallyExpDenom += ISexuallyExpH[ia*5+iy][ig] + ISexuallyExpL[ia*5+iy][ig];
			}
			VirginAgeExitRate[ia][ig] = (IVirginsH[(ia+1)*5][ig] + IVirginsL[(ia+1)*5][ig])/
				VirginDenom;
			if(ig==0){
				AgeExitRateM[ia][0] = (ISexuallyExpH[(ia+1)*5][0] + 
					ISexuallyExpL[(ia+1)*5][0])/SexuallyExpDenom;}
			else{
				AgeExitRateF[ia][0] = (ISexuallyExpH[(ia+1)*5][1] + 
					ISexuallyExpL[(ia+1)*5][1])/SexuallyExpDenom;}
		}
	}

	// Next group together into 5-year age bands
	for(ig=0; ig<2; ig++){
		for(ia=0; ia<16; ia++){
			GVirginsH[ia][ig] = IVirginsH[ia*5][ig];
			GVirginsL[ia][ig] = IVirginsL[ia*5][ig];
			GSexuallyExpH[ia][ig] = ISexuallyExpH[ia*5][ig];
			GSexuallyExpL[ia][ig] = ISexuallyExpL[ia*5][ig];
			GMarriedH[ia][ig] = IMarriedH[ia*5][ig];
			GMarriedL[ia][ig] = IMarriedL[ia*5][ig];
			GSexuallyExpUnmarriedH[ia][ig] = ISexuallyExpUnmarriedH[ia*5][ig];
			GSexuallyExpUnmarriedL[ia][ig] = ISexuallyExpUnmarriedL[ia*5][ig];
			if(ia<15){count = 5;}
			else{count = 6;}
			for(iy=1; iy<count; iy++){
				GVirginsH[ia][ig] += IVirginsH[ia*5+iy][ig];
				GVirginsL[ia][ig] += IVirginsL[ia*5+iy][ig];
				GSexuallyExpH[ia][ig] += ISexuallyExpH[ia*5+iy][ig];
				GSexuallyExpL[ia][ig] += ISexuallyExpL[ia*5+iy][ig];
				GMarriedH[ia][ig] += IMarriedH[ia*5+iy][ig];
				GMarriedL[ia][ig] += IMarriedL[ia*5+iy][ig];
				GSexuallyExpUnmarriedH[ia][ig] += ISexuallyExpUnmarriedH[ia*5+iy][ig];
				GSexuallyExpUnmarriedL[ia][ig] += ISexuallyExpUnmarriedL[ia*5+iy][ig];
			}
		}
	}

	// Then split the cohorts according to numbers of current partners
	for(ig=0; ig<2; ig++){
		for(ia=0; ia<16; ia++){
			cHD = PartnershipFormation[0][ig] * AgeEffectPartners[ia][ig] * MeanDurSTrel[0][0];
			cLD = PartnershipFormation[1][ig] * AgeEffectPartners[ia][ig] * MeanDurSTrel[0][0];
			GUnmarriedH0[ia][ig] = GSexuallyExpUnmarriedH[ia][ig]/(1.0 + cHD + 0.5 *
				PartnerEffectNew[0][ig] * pow(cHD, 2.0));
			GUnmarriedH1[ia][ig] = GSexuallyExpUnmarriedH[ia][ig] * cHD/(1.0 + cHD + 0.5 *
				PartnerEffectNew[0][ig] * pow(cHD, 2.0));
			GUnmarriedH2[ia][ig] = GSexuallyExpUnmarriedH[ia][ig] - GUnmarriedH0[ia][ig] -
				GUnmarriedH1[ia][ig];
			GUnmarriedL0[ia][ig] = GSexuallyExpUnmarriedL[ia][ig]/(1.0 + cLD);
			GUnmarriedL1[ia][ig] = GSexuallyExpUnmarriedL[ia][ig] * cLD/(1.0 + cLD);
			GMarriedH1[ia][ig] = GMarriedH[ia][ig]/(1.0 + cHD * PartnerEffectNew[1][ig]);
			GMarriedH2[ia][ig] = GMarriedH[ia][ig] - GMarriedH1[ia][ig];
		}
	}

	// Next balance male and female demand for ST and LT partners (similar to 
	// BalanceSexualPartners function)
	DesiredSTpartners[0][0] = 0;
	DesiredSTpartners[0][1] = 0;
	DesiredSTpartners[1][0] = 0;
	DesiredSTpartners[1][1] = 0;
	for(ia=0; ia<16; ia++){
		DesiredSTpartners[0][0] += GUnmarriedH1[ia][0] + 2.0 * GUnmarriedH2[ia][0] +
			GMarriedH2[ia][0];
		DesiredSTpartners[0][1] += GUnmarriedH1[ia][1] + 2.0 * GUnmarriedH2[ia][1] +
			GMarriedH2[ia][1];
		DesiredSTpartners[1][0] += GUnmarriedL1[ia][0];
		DesiredSTpartners[1][1] += GUnmarriedL1[ia][1];
	}
		
	DesiredPartnerRiskM[0][0] = (1.0 - AssortativeM) + AssortativeM * 
		DesiredSTpartners[0][1] / (DesiredSTpartners[0][1] + DesiredSTpartners[1][1]);
	DesiredPartnerRiskM[0][1] = 1.0 - DesiredPartnerRiskM[0][0];
	DesiredPartnerRiskM[1][1] = (1.0 - AssortativeM) + AssortativeM * 
		DesiredSTpartners[1][1] / (DesiredSTpartners[0][1] + DesiredSTpartners[1][1]);
	DesiredPartnerRiskM[1][0] = 1.0 - DesiredPartnerRiskM[1][1];
	DesiredPartnerRiskF[0][0] = (1.0 - AssortativeF) + AssortativeF * 
		DesiredSTpartners[0][0] / (DesiredSTpartners[0][0] + DesiredSTpartners[1][0]);
	DesiredPartnerRiskF[0][1] = 1.0 - DesiredPartnerRiskF[0][0];
	DesiredPartnerRiskF[1][1] = (1.0 - AssortativeF) + AssortativeF * 
		DesiredSTpartners[1][0] / (DesiredSTpartners[0][0] + DesiredSTpartners[1][0]);
	DesiredPartnerRiskF[1][0] = 1.0 - DesiredPartnerRiskF[1][1];

	for(ii=0; ii<2; ii++){
		for(ij=0; ij<2; ij++){
			AdjSTrateM[ii][ij] = (GenderEquality * DesiredSTpartners[ij][1] * 
				DesiredPartnerRiskF[ij][ii] + (1.0 - GenderEquality) * 
				DesiredSTpartners[ii][0] * DesiredPartnerRiskM[ii][ij]);
			AdjSTrateF[ii][ij] = (GenderEquality * DesiredSTpartners[ii][1] * 
				DesiredPartnerRiskF[ii][ij] + (1.0 - GenderEquality) * 
				DesiredSTpartners[ij][0] * DesiredPartnerRiskM[ij][ii]);
		}
		ActualPropnSTH[ii][0] = AdjSTrateM[ii][0]/(AdjSTrateM[ii][0] + AdjSTrateM[ii][1]);
		ActualPropnSTH[ii][1] = AdjSTrateF[ii][0]/(AdjSTrateF[ii][0] + AdjSTrateF[ii][1]);
	}

	DesiredMarriagesM[0][0] = ((1.0 - AssortativeM) + AssortativeM * HighPropnF) * HighPropnM;
	DesiredMarriagesM[0][1] = AssortativeM * (1.0 - HighPropnF) * HighPropnM;
	DesiredMarriagesM[1][1] = ((1.0 - AssortativeM) + AssortativeM * (1.0 - HighPropnF)) *
		(1.0 - HighPropnM);
	DesiredMarriagesM[1][0] = AssortativeM * HighPropnF * (1 - HighPropnM);
	DesiredMarriagesF[0][0] = ((1.0 - AssortativeF) + AssortativeF * HighPropnM) * HighPropnF;
	DesiredMarriagesF[0][1] = AssortativeF * (1.0 - HighPropnM) * HighPropnF;
	DesiredMarriagesF[1][1] = ((1.0 - AssortativeF) + AssortativeF * (1.0 - HighPropnM)) *
		(1.0 - HighPropnF);
	DesiredMarriagesF[1][0] = AssortativeF * HighPropnM * (1 - HighPropnF);

	for(ii=0; ii<2; ii++){
		for(ij=0; ij<2; ij++){
			AdjLTrateM[ii][ij] = (GenderEquality * DesiredMarriagesF[ij][ii] +
				(1.0 - GenderEquality) * DesiredMarriagesM[ii][ij]);
			AdjLTrateF[ii][ij] = (GenderEquality * DesiredMarriagesF[ii][ij] + 
				(1.0 - GenderEquality) * DesiredMarriagesM[ij][ii]);
		}
		ActualPropnLTH[ii][0] = AdjLTrateM[ii][0]/(AdjLTrateM[ii][0] + AdjLTrateM[ii][1]);
		ActualPropnLTH[ii][1] = AdjLTrateF[ii][0]/(AdjLTrateF[ii][0] + AdjLTrateF[ii][1]);
	}

	// Calculate the sex work parameters and numbers of sex workers
	lambda = GammaMeanFSW/pow(GammaStdDevFSW, 2.0);
	alpha = GammaMeanFSW * lambda;
	for(ia=0; ia<16; ia++){
		AgeEffectFSWcontact[ia] = pow(lambda, alpha) * pow(ia*5.0 + 2.5, alpha - 1.0) *
			exp(-lambda * (ia*5.0 + 2.5));
		FSWcontactBase[ia] = AgeEffectFSWcontact[ia] * (GUnmarriedH0[ia][0] * 
			PartnerEffectFSWcontact[0] + GUnmarriedH1[ia][0] * PartnerEffectFSWcontact[1] + 
			GMarriedH1[ia][0] * PartnerEffectFSWcontact[2] + GUnmarriedH2[ia][0] * 
			PartnerEffectFSWcontact[3] + GMarriedH2[ia][0] * PartnerEffectFSWcontact[4]);
	}
	BaseFSWdemand = 0;
	AdjFSWdemand = 0;
	MalePop15to49 = 0;
	for(ia=1; ia<8; ia++){
		BaseFSWdemand += FSWcontactBase[ia];
		MalePop15to49 += GSexuallyExpH[ia][0] + GSexuallyExpL[ia][0] + GVirginsH[ia][0] +
			GVirginsL[ia][0];
	}
	FSWcontactConstant = MeanFSWcontacts * MalePop15to49/BaseFSWdemand;
	for(ia=0; ia<16; ia++){
		FSWcontactDemand[ia] = FSWcontactBase[ia] * FSWcontactConstant;
		AdjFSWdemand += FSWcontactDemand[ia];
	}

	// Calculate the number of individuals at each age, in each sexual behaviour cohort
	for(ia=0; ia<16; ia++){
		MaleHigh.Virgin.TotalAlive[ia] = GVirginsH[ia][0];
		MaleHigh.NoPartner.TotalAlive[ia] = GUnmarriedH0[ia][0];
		MaleHigh.S1.TotalAlive[ia] = GUnmarriedH1[ia][0] * ActualPropnSTH[0][0];
		MaleHigh.S2.TotalAlive[ia] = GUnmarriedH1[ia][0] * (1.0 - ActualPropnSTH[0][0]);
		MaleHigh.L1.TotalAlive[ia] = GMarriedH1[ia][0] * ActualPropnLTH[0][0];
		MaleHigh.L2.TotalAlive[ia] = GMarriedH1[ia][0] * (1.0 - ActualPropnLTH[0][0]);
		MaleHigh.S11.TotalAlive[ia] = GUnmarriedH2[ia][0] * pow(ActualPropnSTH[0][0], 2.0);
		MaleHigh.S12.TotalAlive[ia] = GUnmarriedH2[ia][0] * 2.0 * ActualPropnSTH[0][0] *
			(1.0 - ActualPropnSTH[0][0]);
		MaleHigh.S22.TotalAlive[ia] = GUnmarriedH2[ia][0] * pow(1.0 - ActualPropnSTH[0][0], 2.0);
		MaleHigh.L11.TotalAlive[ia] = GMarriedH2[ia][0] * ActualPropnLTH[0][0] *
			ActualPropnSTH[0][0];
		MaleHigh.L12.TotalAlive[ia] = GMarriedH2[ia][0] * ActualPropnLTH[0][0] *
			(1.0 - ActualPropnSTH[0][0]);
		MaleHigh.L21.TotalAlive[ia] = GMarriedH2[ia][0] * (1.0 - ActualPropnLTH[0][0]) *
			ActualPropnSTH[0][0];
		MaleHigh.L22.TotalAlive[ia] = GMarriedH2[ia][0] * (1.0 - ActualPropnLTH[0][0]) *
			(1.0 - ActualPropnSTH[0][0]);
		MaleLow.Virgin.TotalAlive[ia] = GVirginsL[ia][0];
		MaleLow.NoPartner.TotalAlive[ia] = GUnmarriedL0[ia][0];
		MaleLow.S1.TotalAlive[ia] = GUnmarriedL1[ia][0] * ActualPropnSTH[1][0];
		MaleLow.S2.TotalAlive[ia] = GUnmarriedL1[ia][0] * (1.0 - ActualPropnSTH[1][0]);
		MaleLow.L1.TotalAlive[ia] = GMarriedL[ia][0] * ActualPropnLTH[1][0];
		MaleLow.L2.TotalAlive[ia] = GMarriedL[ia][0] * (1.0 - ActualPropnLTH[1][0]);

		FemHigh.Virgin.TotalAlive[ia] = GVirginsH[ia][1];
		FemHigh.FSW.TotalAlive[ia] = InitFSWageDbn[ia] * AdjFSWdemand/AnnNumberClients;
		FemHigh.NoPartner.TotalAlive[ia] = GUnmarriedH0[ia][1] - FemHigh.FSW.TotalAlive[ia];
		if(FemHigh.NoPartner.TotalAlive[ia]<0){
			FemHigh.NoPartner.TotalAlive[ia] = 0.0;
			ErrorInd = 1;
			ErrorCount += 1;
		}
		FemHigh.S1.TotalAlive[ia] = GUnmarriedH1[ia][1] * ActualPropnSTH[0][1];
		FemHigh.S2.TotalAlive[ia] = GUnmarriedH1[ia][1] * (1.0 - ActualPropnSTH[0][1]);
		FemHigh.L1.TotalAlive[ia] = GMarriedH1[ia][1] * ActualPropnLTH[0][1];
		FemHigh.L2.TotalAlive[ia] = GMarriedH1[ia][1] * (1.0 - ActualPropnLTH[0][1]);
		FemHigh.S11.TotalAlive[ia] = GUnmarriedH2[ia][1] * pow(ActualPropnSTH[0][1], 2.0);
		FemHigh.S12.TotalAlive[ia] = GUnmarriedH2[ia][1] * 2.0 * ActualPropnSTH[0][1] *
			(1.0 - ActualPropnSTH[0][1]);
		FemHigh.S22.TotalAlive[ia] = GUnmarriedH2[ia][1] * pow(1.0 - ActualPropnSTH[0][1], 2.0);
		FemHigh.L11.TotalAlive[ia] = GMarriedH2[ia][1] * ActualPropnLTH[0][1] *
			ActualPropnSTH[0][1];
		FemHigh.L12.TotalAlive[ia] = GMarriedH2[ia][1] * ActualPropnLTH[0][1] *
			(1.0 - ActualPropnSTH[0][1]);
		FemHigh.L21.TotalAlive[ia] = GMarriedH2[ia][1] * (1.0 - ActualPropnLTH[0][1]) *
			ActualPropnSTH[0][1];
		FemHigh.L22.TotalAlive[ia] = GMarriedH2[ia][1] * (1.0 - ActualPropnLTH[0][1]) *
			(1.0 - ActualPropnSTH[0][1]);
		FemLow.Virgin.TotalAlive[ia] = GVirginsL[ia][1];
		FemLow.NoPartner.TotalAlive[ia] = GUnmarriedL0[ia][1];
		FemLow.S1.TotalAlive[ia] = GUnmarriedL1[ia][1] * ActualPropnSTH[1][1];
		FemLow.S2.TotalAlive[ia] = GUnmarriedL1[ia][1] * (1.0 - ActualPropnSTH[1][1]);
		FemLow.L1.TotalAlive[ia] = GMarriedL[ia][1] * ActualPropnLTH[1][1];
		FemLow.L2.TotalAlive[ia] = GMarriedL[ia][1] * (1.0 - ActualPropnLTH[1][1]);
	}

	// Finally, calculate the rates of entry into the FSW group
	// (We're implicitly assuming that within each age band the ages of sex workers are
	// uniformly distributed, so that 20% of sex workers 'age out' of age band at year end.)
	FSWentry[0] = FemHigh.FSW.TotalAlive[0] * (0.2 + FSWexit[0])/
		FemHigh.NoPartner.TotalAlive[0];
	for(ia=1; ia<16; ia++){
		FSWentry[ia] = (FemHigh.FSW.TotalAlive[ia] * (0.2 + FSWexit[ia]) - 0.2 *
			FemHigh.FSW.TotalAlive[ia-1])/FemHigh.NoPartner.TotalAlive[ia];
		if(FSWentry[ia]<0){FSWentry[ia] = 0;}
	}
}

void ResetAll()
{
	ReadAllInputFiles();
	GetAllPartnerRates();
	GetStartProfile();
	MaleHigh.Reset();
	MaleLow.Reset();
	FemHigh.Reset();
	FemLow.Reset();
	MaleChild.Reset();
	FemChild.Reset();
	GetAllNumbersBySTDstage();
	GetSummary();
}

void OneYearProj()
{
	int ic;

	//HAARTaccess[CurrYear-StartYear] = 0.0;
	UpdateFertAndBirths();
	CalcPrevForLogL();
	UpdateProbCure();
	UpdateSTDtransitionProbs();
	//if(CurrYear==StartYear){UpdateCondomUse();}
	UpdateCondomUse();
	if(SexCalib==1){CalcBehavForSumSq();}
	UpdateNonAIDSmort();

	for(ic=0; ic<CycleS; ic++){
		OneBehavCycle();}

	CalcAllAgeChanges();
	MaleHigh.GetAllPropnsBySTDstage();
	MaleLow.GetAllPropnsBySTDstage();
	FemHigh.GetAllPropnsBySTDstage();
	FemLow.GetAllPropnsBySTDstage();

	GetSummary();
}

void UpdateFertAndBirths()
{
	int ia, is, yr;

	yr = CurrYear - StartYear;

	// Calculate births to women in each age group
	for(ia=0; ia<7; ia++){
		HIVnegFert[ia] = FertilityTable[ia][yr];
		SexuallyExpFert[ia] = HIVnegFert[ia]/(1.0 - VirginsSum[ia+1][1]/TotalPopSum[ia+1][1]);
		if(HIVind==1){
			BirthsToHIVmothers[ia] = 0;
			for(is=0; is<5; is++){
				BirthsToHIVmothers[ia] += HIVstageSumF[ia+1][is+1] * SexuallyExpFert[ia] *
					RelHIVfertility[is];
			}
		}
		BirthsByAge[ia] = (HIVstageSumF[ia+1][0] - VirginsSum[ia+1][1]) * SexuallyExpFert[ia];
		if(HIVind==1){
			BirthsByAge[ia] += BirthsToHIVmothers[ia];}
	}

	// Sum across age groups
	TotBirthsToHIVmothers = 0;
	TotalBirths = 0;
	for(ia=0; ia<7; ia++){
		TotBirthsToHIVmothers += BirthsToHIVmothers[ia];
		TotalBirths += BirthsByAge[ia];
	}

	// Calculate new HIV infections in infants and assign to MaleChild & FemChild
	if(HIVind==1){
		NewHIVperinatal = TotBirthsToHIVmothers * PropnInfectedAtBirth * (1.0 - 
			PMTCTaccess[yr] * AcceptScreening * AcceptNVP * RednNVP);
		NewHIVbreastmilk = (TotBirthsToHIVmothers - NewHIVperinatal) * PropnInfectedAfterBirth
			* (1.0 - PMTCTaccess[yr] * AcceptScreening * RednFF);
		MaleChild.UpdateBirths();
		FemChild.UpdateBirths();
	}
}

void CalcPrevForLogL()
{
	// This function calculates all the model estimates of HIV and STD prevalence necessary
	// for the calculation of the log likelihood, in the current projection year. This should
	// only get called AFTER the UpdateFertAndBirths function, since this function generates
	// the inputs necessary for the calculation of antenatal prevalence.

	int ia; 
	double numerator, denominator, temp1, temp2;

	if(HSVcalib==1 || HDcalib==1){
		UpdateSyndromePropns();
		MaleHigh.GetTotalGUD();
		MaleLow.GetTotalGUD();
		FemHigh.GetTotalGUD();
		FemLow.GetTotalGUD();
		TotalGUDcases[0] = MaleHigh.TotalGUDcases + MaleLow.TotalGUDcases;
		TotalGUDcases[1] = FemHigh.TotalGUDcases + FemLow.TotalGUDcases;
	}
	if(HDcalib==1){
		HDtransitionF.GetGUDprev();
		HDtransitionM.GetGUDprev();
	}
	if(HIVcalib==1){
		HIVtransitionF.CSWprevalence = FemHigh.FSW.ReturnHIVprev();
		HIVtransitionF.GetANCprev();
		HIVtransitionF.GetCSWprev();
		HIVtransitionF.GetHHprev();
		HIVtransitionM.GetHHprev();
		// Generate outputs
		HIVprevFSW[CurrYear-StartYear] = HIVtransitionF.CSWprevalence;
		numerator = 0;
		denominator = 0;
		for(ia=1; ia<8; ia++){
			numerator += HIVstageSumM[ia][0];
			denominator += TotalPopSum[ia][0];
		}
		HIVprev15to49M[CurrYear-StartYear] = 1.0 - numerator/denominator;
		numerator = 0;
		denominator = 0;
		for(ia=1; ia<8; ia++){
			numerator += HIVstageSumF[ia][0];
			denominator += TotalPopSum[ia][1];
		}
		HIVprev15to49F[CurrYear-StartYear] = 1.0 - numerator/denominator;
		if(FixedUncertainty==1){
			temp1 = 0.0;
			temp2 = 0.0;
			for(ia=0; ia<16; ia++){
				temp1 += TotalPopSum[ia][0] + TotalPopSum[ia][1];
				temp2 += HIVstageSumM[ia][0] + HIVstageSumF[ia][0];
			}
			for(ia=0; ia<15; ia++){
				temp1 += MaleChild.TotalAlive[ia] + FemChild.TotalAlive[ia];}
			for(ia=0; ia<10; ia++){
				temp2 += MaleChild.HIVneg[ia] + FemChild.HIVneg[ia];}
			OutTotalPop.out[CurrSim-1][CurrYear-StartYear] = temp1;
			OutTotalHIV.out[CurrSim-1][CurrYear-StartYear] = temp1 - temp2;
			//Out15to49prevM.out[CurrSim-1][CurrYear-StartYear] = HIVprev15to49M[CurrYear-StartYear];
			//Out15to49prevF.out[CurrSim-1][CurrYear-StartYear] = HIVprev15to49F[CurrYear-StartYear];
			//OutCSWprev.out[CurrSim-1][CurrYear-StartYear] = HIVprevFSW[CurrYear-StartYear];
			OutANCprevTot.out[CurrSim-1][CurrYear-StartYear] = TotBirthsToHIVmothers/
				TotalBirths; // Note that we are not yet adjusting for bias
			if(CurrYear>=1990 && CurrYear<=2008){
				OutANCprev15.out[CurrSim-1][CurrYear-1990] = BirthsToHIVmothers[0]/BirthsByAge[0];
				OutANCprev20.out[CurrSim-1][CurrYear-1990] = BirthsToHIVmothers[1]/BirthsByAge[1];
				OutANCprev25.out[CurrSim-1][CurrYear-1990] = BirthsToHIVmothers[2]/BirthsByAge[2];
				OutANCprev30.out[CurrSim-1][CurrYear-1990] = BirthsToHIVmothers[3]/BirthsByAge[3];
				OutANCprev35.out[CurrSim-1][CurrYear-1990] = BirthsToHIVmothers[4]/BirthsByAge[4];
			}
			if(CurrYear==2002){
				for(ia=0; ia<8; ia++){
					OutHSRC2002M.out[CurrSim-1][ia] = 1.0 - HIVstageSumM[ia+1][0]/
						TotalPopSum[ia+1][0];
					OutHSRC2002F.out[CurrSim-1][ia] = 1.0 - HIVstageSumF[ia+1][0]/
						TotalPopSum[ia+1][1];
				}
			}
			/*if(CurrYear==2003){
				for(ia=0; ia<2; ia++){
					OutRHRUprev.out[CurrSim-1][ia] = 1.0 - HIVstageSumM[ia+1][0]/
						TotalPopSum[ia+1][0];
					OutRHRUprev.out[CurrSim-1][ia+2] = 1.0 - HIVstageSumF[ia+1][0]/
						TotalPopSum[ia+1][1];
				}
			}*/
		}
	}
	if(NGcalib==1){
		FemHigh.FSW.GetCSWprev(&NGtransitionF, &FemHigh.FSW.FNG, 0);
		NGtransitionF.GetANCprev();
		NGtransitionF.GetFPCprev();
		NGtransitionF.GetCSWprev();
		NGtransitionF.GetHHprev();
		NGtransitionM.GetHHprev();
		// Calculate outputs
		NGprevFSW[CurrYear-StartYear] = NGtransitionF.CSWprevalence;
		numerator = 0;
		denominator = 0;
		for(ia=1; ia<8; ia++){
			numerator += NGtransitionM.AliveSum[ia][0];
			denominator += TotalPopSum[ia][0];
		}
		NGprev15to49M[CurrYear-StartYear] = 1.0 - numerator/denominator;
		numerator = 0;
		denominator = 0;
		for(ia=1; ia<8; ia++){
			numerator += NGtransitionF.AliveSum[ia][0];
			denominator += TotalPopSum[ia][1];
		}
		NGprev15to49F[CurrYear-StartYear] = 1.0 - numerator/denominator;
	}
}

void UpdateProbCure()
{
	if(HSVind==1){
		HSVtransitionM.CalcProbCure();
		HSVtransitionF.CalcProbCure();}
	if(TPind==1){
		TPtransitionM.CalcProbCure();
		TPtransitionF.CalcProbCure();}
	if(HDind==1){
		HDtransitionM.CalcProbCure();
		HDtransitionF.CalcProbCure();}
	if(NGind==1){
		NGtransitionM.CalcProbCure();
		NGtransitionF.CalcProbCure();}
	if(CTind==1){
		CTtransitionM.CalcProbCure();
		CTtransitionF.CalcProbCure();}
	if(TVind==1){
		TVtransitionM.CalcProbCure();
		TVtransitionF.CalcProbCure();}
	if(BVind==1){
		BVtransitionF.CalcProbCure();
		BVtransitionF.CalcProbPartialCure();}
	if(VCind==1){
		VCtransitionF.CalcProbCure();
		VCtransitionF.CalcProbPartialCure();}
}

void UpdateSTDtransitionProbs()
{
	if(HIVind==1){
		HIVtransitionM.CalcTransitionProbs();
		HIVtransitionF.CalcTransitionProbs();}
	if(HSVind==1){
		HSVtransitionM.CalcTransitionProbs();
		HSVtransitionF.CalcTransitionProbs();}
	if(TPind==1){
		TPtransitionM.CalcTransitionProbs();
		TPtransitionF.CalcTransitionProbs();}
	if(HDind==1){
		HDtransitionM.CalcTransitionProbs();
		HDtransitionF.CalcTransitionProbs();}
	if(NGind==1){
		NGtransitionM.CalcTransitionProbs();
		NGtransitionF.CalcTransitionProbs();}
	if(CTind==1){
		CTtransitionM.CalcTransitionProbs();
		CTtransitionF.CalcTransitionProbs();}
	if(TVind==1){
		TVtransitionM.CalcTransitionProbs();
		TVtransitionF.CalcTransitionProbs();}
	if(BVind==1){
		BVtransitionF.CalcTransitionProbs();}
	if(VCind==1){
		VCtransitionF.CalcTransitionProbs();}
}

void UpdateCondomUse()
{
	int ia, ib;
	double Rate15to19[3], x;

	BaselineCondomUse = 0.08 + CondomScaling * (0.2 - 0.08);
	RatioUltTo1998[0] = 3.0 + CondomScaling * (15.0 - 3.0);
	RatioUltTo1998[1] = 1.5 + CondomScaling * (7.0 - 1.5);
	ShapeBehavChange[0] = 3.8 + CondomScaling * (2.8 - 3.8);
	ShapeBehavChange[1] = 3.6 + CondomScaling * (1.8 - 3.6);
	MedianToBehavChange[0] = 13.0 * exp(-log(log(1.0 - log(RatioInitialTo1998[0])/
		log(RatioUltTo1998[0]))/log(2.0))/ShapeBehavChange[0]);
	MedianToBehavChange[1] = 13.0 * exp(-log(log(1.0 - log(RatioInitialTo1998[1])/
		log(RatioUltTo1998[1]))/log(2.0))/ShapeBehavChange[1]);
	
	for(ib=0; ib<3; ib++){
		x = (BaselineCondomUse/(1.0 - BaselineCondomUse)) * RelEffectCondom[ib] *
			RatioInitialTo1998[ib] * pow(RatioUltTo1998[ib]/RatioInitialTo1998[ib], 1.0 - 
			pow(0.5, pow((CurrYear - 1985)/MedianToBehavChange[ib], ShapeBehavChange[ib])));
		Rate15to19[ib] = x/(1.0 + x);
	}
	// Condom usage for females in ST relationships
	for(ia=0; ia<16; ia++){
		x = (Rate15to19[0]/(1.0 - Rate15to19[0])) * exp(5.0 * (ia - 1) * AgeEffectCondom[0]);
		CondomUseST[ia][1] = x/(1.0 + x);
	}
	// Condom usage for females in LT relationships
	for(ia=0; ia<16; ia++){
		x = (Rate15to19[1]/(1.0 - Rate15to19[1])) * exp(5.0 * (ia - 1) * AgeEffectCondom[1]);
		CondomUseLT[ia][1] = x/(1.0 + x);
	}
	// Condom use in males
	for(ia=0; ia<16; ia++){
		CondomUseST[ia][0] = 0;
		CondomUseLT[ia][0] = 0;
		for(ib=0; ib<16; ib++){
			CondomUseST[ia][0] += AgePrefM[ia][ib] * CondomUseST[ib][1];
			CondomUseLT[ia][0] += AgePrefM[ia][ib] * CondomUseLT[ib][1];
		}
	}
	// Condom use in FSW-client relationships
	CondomUseFSW = Rate15to19[2];
}

void CalcBehavForSumSq()
{
	if(CurrYear==1996){GetMarriageOutput1996();}
	if(CurrYear==2001){GetMarriageOutput2001();}
	if(CurrYear==2007){GetMarriageOutput2007();}
	if(CurrYear==2005){GetPartnerOutput2005();}
	
	PropnActsProtectedM[CurrYear-StartYear] = (UnmarriedActiveSum[1][0] * 
		CondomUseST[1][0] + MarriedSum[1][0] * CondomUseLT[1][0] + UnmarriedActiveSum[2][0] * 
		CondomUseST[2][0] + MarriedSum[2][0] * CondomUseLT[2][0])/(UnmarriedActiveSum[1][0] + 
		MarriedSum[1][0] + UnmarriedActiveSum[2][0] + MarriedSum[2][0]);
	PropnActsProtectedF[CurrYear-StartYear] = (UnmarriedActiveSum[1][1] * 
		CondomUseST[1][1] + MarriedSum[1][1] * CondomUseLT[1][1] + UnmarriedActiveSum[2][1] * 
		CondomUseST[2][1] + MarriedSum[2][1] * CondomUseLT[2][1])/(UnmarriedActiveSum[1][1] + 
		MarriedSum[1][1] + UnmarriedActiveSum[2][1] + MarriedSum[2][1]);
}

void UpdateNonAIDSmort()
{
	int ia, ib;

	for(ia=0; ia<16; ia++){
		NonAIDSmortForce[ia][0] = -log(1.0 - NonAIDSmortM[ia][CurrYear-1985]);
		NonAIDSmortForce[ia][1] = -log(1.0 - NonAIDSmortF[ia][CurrYear-1985]);
		NonAIDSmortProb[ia][0] = 1.0 - exp(-NonAIDSmortForce[ia][0]/CycleS);
		NonAIDSmortProb[ia][1] = 1.0 - exp(-NonAIDSmortForce[ia][1]/CycleS);
		NonAIDSmortPartner[ia][0] = 0;
		NonAIDSmortPartner[ia][1] = 0;
	}
	for(ia=0; ia<16; ia++){
		for(ib=0; ib<16; ib++){
			NonAIDSmortPartner[ia][0] += AgePrefM[ia][ib] * NonAIDSmortForce[ib][1];
			NonAIDSmortPartner[ia][1] += AgePrefF[ia][ib] * NonAIDSmortForce[ib][0];
		}
	}
	MaleChild.UpdateMort();
	FemChild.UpdateMort();
}

void CalcAllAgeChanges()
{
	MaleHigh.CalcAllAgeChanges();
	MaleLow.CalcAllAgeChanges();
	FemHigh.CalcAllAgeChanges();
	FemLow.CalcAllAgeChanges();

	MaleHigh.GetNewVirgins();
	MaleLow.GetNewVirgins();
	FemHigh.GetNewVirgins();
	FemLow.GetNewVirgins();

	MaleChild.CalcAgeChanges();
	FemChild.CalcAgeChanges();
}

void GetSummary()
{
	int ia, is, states;
	double temp1, temp2, mean, var;
	double NegAtStart[16][2];

	// Generate numbers of HIV-negatives at START of year (all other outputs are calculated at year end)
	// This is needed for the calculation of HIV incidence rates later.
	for(ia=0; ia<16; ia++){
		NegAtStart[ia][0] = HIVstageSumM[ia][0];
		NegAtStart[ia][1] = HIVstageSumF[ia][0];
	}

	// Generate outputs in the 'Pop profile' sheet
	for(ia=0; ia<16; ia++){
		TotalPopSum[ia][0] = MaleHigh.Virgin.TotalAlive[ia] + MaleHigh.NoPartner.TotalAlive[ia] +
			MaleHigh.S1.TotalAlive[ia] + MaleHigh.S2.TotalAlive[ia] + MaleHigh.L1.TotalAlive[ia] +
			MaleHigh.L2.TotalAlive[ia] + MaleHigh.S11.TotalAlive[ia] +
			MaleHigh.S12.TotalAlive[ia] + MaleHigh.S22.TotalAlive[ia] +
			MaleHigh.L11.TotalAlive[ia] + MaleHigh.L12.TotalAlive[ia] +
			MaleHigh.L21.TotalAlive[ia] + MaleHigh.L22.TotalAlive[ia] +
			MaleLow.Virgin.TotalAlive[ia] + MaleLow.NoPartner.TotalAlive[ia] +
			MaleLow.S1.TotalAlive[ia] + MaleLow.S2.TotalAlive[ia] + MaleLow.L1.TotalAlive[ia] +
			MaleLow.L2.TotalAlive[ia];
		TotalPopSum[ia][1] = FemHigh.Virgin.TotalAlive[ia] + FemHigh.NoPartner.TotalAlive[ia] +
			FemHigh.S1.TotalAlive[ia] + FemHigh.S2.TotalAlive[ia] + FemHigh.L1.TotalAlive[ia] +
			FemHigh.L2.TotalAlive[ia] + FemHigh.S11.TotalAlive[ia] +
			FemHigh.S12.TotalAlive[ia] + FemHigh.S22.TotalAlive[ia] +
			FemHigh.L11.TotalAlive[ia] + FemHigh.L12.TotalAlive[ia] +
			FemHigh.L21.TotalAlive[ia] + FemHigh.L22.TotalAlive[ia] +
			FemLow.Virgin.TotalAlive[ia] + FemLow.NoPartner.TotalAlive[ia] +
			FemLow.S1.TotalAlive[ia] + FemLow.S2.TotalAlive[ia] + FemLow.L1.TotalAlive[ia] +
			FemLow.L2.TotalAlive[ia] + FemHigh.FSW.TotalAlive[ia];
		VirginsSum[ia][0] = MaleHigh.Virgin.TotalAlive[ia] + MaleLow.Virgin.TotalAlive[ia];
		VirginsSum[ia][1] = FemHigh.Virgin.TotalAlive[ia] + FemLow.Virgin.TotalAlive[ia];
		MarriedSum[ia][0] = MaleHigh.L1.TotalAlive[ia] + MaleHigh.L2.TotalAlive[ia] +
			MaleHigh.L11.TotalAlive[ia] + MaleHigh.L12.TotalAlive[ia] +
			MaleHigh.L21.TotalAlive[ia] + MaleHigh.L22.TotalAlive[ia] +
			MaleLow.L1.TotalAlive[ia] + MaleLow.L2.TotalAlive[ia];
		MarriedSum[ia][1] = FemHigh.L1.TotalAlive[ia] + FemHigh.L2.TotalAlive[ia] +
			FemHigh.L11.TotalAlive[ia] + FemHigh.L12.TotalAlive[ia] +
			FemHigh.L21.TotalAlive[ia] + FemHigh.L22.TotalAlive[ia] +
			FemLow.L1.TotalAlive[ia] + FemLow.L2.TotalAlive[ia];
		UnmarriedActiveSum[ia][0] = MaleHigh.S1.TotalAlive[ia] + MaleHigh.S2.TotalAlive[ia] +
			MaleHigh.S11.TotalAlive[ia] + MaleHigh.S12.TotalAlive[ia] +
			MaleHigh.S22.TotalAlive[ia] + MaleLow.S1.TotalAlive[ia] + 
			MaleLow.S2.TotalAlive[ia];
		UnmarriedActiveSum[ia][1] = FemHigh.S1.TotalAlive[ia] + FemHigh.S2.TotalAlive[ia] +
			FemHigh.S11.TotalAlive[ia] + FemHigh.S12.TotalAlive[ia] +
			FemHigh.S22.TotalAlive[ia] + FemLow.S1.TotalAlive[ia] + 
			FemLow.S2.TotalAlive[ia];
		UnmarriedMultSum[ia][0] = MaleHigh.S11.TotalAlive[ia] + MaleHigh.S12.TotalAlive[ia] +
			MaleHigh.S22.TotalAlive[ia];
		UnmarriedMultSum[ia][1] = FemHigh.S11.TotalAlive[ia] + FemHigh.S12.TotalAlive[ia] +
			FemHigh.S22.TotalAlive[ia];
		MultPartnerSum[ia][0] = MaleHigh.S11.TotalAlive[ia] +
			MaleHigh.S12.TotalAlive[ia] + MaleHigh.S22.TotalAlive[ia] +
			MaleHigh.L11.TotalAlive[ia] + MaleHigh.L12.TotalAlive[ia] +
			MaleHigh.L21.TotalAlive[ia] + MaleHigh.L22.TotalAlive[ia];
		MultPartnerSum[ia][1] = FemHigh.S11.TotalAlive[ia] +
			FemHigh.S12.TotalAlive[ia] + FemHigh.S22.TotalAlive[ia] +
			FemHigh.L11.TotalAlive[ia] + FemHigh.L12.TotalAlive[ia] +
			FemHigh.L21.TotalAlive[ia] + FemHigh.L22.TotalAlive[ia];
		LowRiskSum[ia][0] = MaleLow.Virgin.TotalAlive[ia] + MaleLow.NoPartner.TotalAlive[ia] +
			MaleLow.S1.TotalAlive[ia] + MaleLow.S2.TotalAlive[ia] + MaleLow.L1.TotalAlive[ia] +
			MaleLow.L2.TotalAlive[ia];
		LowRiskSum[ia][1] = FemLow.Virgin.TotalAlive[ia] + FemLow.NoPartner.TotalAlive[ia] +
			FemLow.S1.TotalAlive[ia] + FemLow.S2.TotalAlive[ia] + FemLow.L1.TotalAlive[ia] +
			FemLow.L2.TotalAlive[ia];

		// Generate outputs in the 'STD profile' sheet
		if(HIVind==1){
			states = 6;}
		else{
			states = 1;}
		for(is=0; is<states; is++){
			HIVstageSumM[ia][is] = MaleHigh.Virgin.NumbersByHIVstage[ia][is] + 
				MaleHigh.NoPartner.NumbersByHIVstage[ia][is] + MaleHigh.S1.NumbersByHIVstage[ia][is] + 
				MaleHigh.S2.NumbersByHIVstage[ia][is] + MaleHigh.L1.NumbersByHIVstage[ia][is] +
				MaleHigh.L2.NumbersByHIVstage[ia][is] + MaleHigh.S11.NumbersByHIVstage[ia][is] +
				MaleHigh.S12.NumbersByHIVstage[ia][is] + MaleHigh.S22.NumbersByHIVstage[ia][is] +
				MaleHigh.L11.NumbersByHIVstage[ia][is] + MaleHigh.L12.NumbersByHIVstage[ia][is] +
				MaleHigh.L21.NumbersByHIVstage[ia][is] + MaleHigh.L22.NumbersByHIVstage[ia][is] +
				MaleLow.Virgin.NumbersByHIVstage[ia][is] + MaleLow.NoPartner.NumbersByHIVstage[ia][is] +
				MaleLow.S1.NumbersByHIVstage[ia][is] + MaleLow.S2.NumbersByHIVstage[ia][is] + 
				MaleLow.L1.NumbersByHIVstage[ia][is] + MaleLow.L2.NumbersByHIVstage[ia][is];
			HIVstageSumF[ia][is] = FemHigh.Virgin.NumbersByHIVstage[ia][is] + 
				FemHigh.NoPartner.NumbersByHIVstage[ia][is] + FemHigh.S1.NumbersByHIVstage[ia][is] + 
				FemHigh.S2.NumbersByHIVstage[ia][is] + FemHigh.L1.NumbersByHIVstage[ia][is] +
				FemHigh.L2.NumbersByHIVstage[ia][is] + FemHigh.S11.NumbersByHIVstage[ia][is] +
				FemHigh.S12.NumbersByHIVstage[ia][is] + FemHigh.S22.NumbersByHIVstage[ia][is] +
				FemHigh.L11.NumbersByHIVstage[ia][is] + FemHigh.L12.NumbersByHIVstage[ia][is] +
				FemHigh.L21.NumbersByHIVstage[ia][is] + FemHigh.L22.NumbersByHIVstage[ia][is] +
				FemLow.Virgin.NumbersByHIVstage[ia][is] + FemLow.NoPartner.NumbersByHIVstage[ia][is] +
				FemLow.S1.NumbersByHIVstage[ia][is] + FemLow.S2.NumbersByHIVstage[ia][is] + 
				FemLow.L1.NumbersByHIVstage[ia][is] + FemLow.L2.NumbersByHIVstage[ia][is] +
				FemHigh.FSW.NumbersByHIVstage[ia][is];
		}
		/*if(CurrYear==2006){
			OutHIVnegUM.out[CurrSim-1][ia] = MaleHigh.Virgin.NumbersByHIVstage[ia][0] + 
				MaleHigh.NoPartner.NumbersByHIVstage[ia][0] + MaleHigh.S1.NumbersByHIVstage[ia][0] + 
				MaleHigh.S2.NumbersByHIVstage[ia][0] + MaleHigh.S11.NumbersByHIVstage[ia][0] +
				MaleHigh.S12.NumbersByHIVstage[ia][0] + MaleHigh.S22.NumbersByHIVstage[ia][0] +
				MaleLow.Virgin.NumbersByHIVstage[ia][0] + MaleLow.NoPartner.NumbersByHIVstage[ia][0] +
				MaleLow.S1.NumbersByHIVstage[ia][0] + MaleLow.S2.NumbersByHIVstage[ia][0];
			OutHIVnegMM.out[CurrSim-1][ia] = MaleHigh.L1.NumbersByHIVstage[ia][0] +
				MaleHigh.L2.NumbersByHIVstage[ia][0] + 
				MaleHigh.L11.NumbersByHIVstage[ia][0] + MaleHigh.L12.NumbersByHIVstage[ia][0] +
				MaleHigh.L21.NumbersByHIVstage[ia][0] + MaleHigh.L22.NumbersByHIVstage[ia][0] +
				MaleLow.L1.NumbersByHIVstage[ia][0] + MaleLow.L2.NumbersByHIVstage[ia][0];
			OutHIVnegUF.out[CurrSim-1][ia] = FemHigh.Virgin.NumbersByHIVstage[ia][0] + 
				FemHigh.NoPartner.NumbersByHIVstage[ia][0] + FemHigh.S1.NumbersByHIVstage[ia][0] + 
				FemHigh.S2.NumbersByHIVstage[ia][0] + FemHigh.S11.NumbersByHIVstage[ia][0] +
				FemHigh.S12.NumbersByHIVstage[ia][0] + FemHigh.S22.NumbersByHIVstage[ia][0] +
				FemLow.Virgin.NumbersByHIVstage[ia][0] + FemLow.NoPartner.NumbersByHIVstage[ia][0] +
				FemLow.S1.NumbersByHIVstage[ia][0] + FemLow.S2.NumbersByHIVstage[ia][0] + 
				FemHigh.FSW.NumbersByHIVstage[ia][0];
			OutHIVnegMF.out[CurrSim-1][ia] = FemHigh.L1.NumbersByHIVstage[ia][0] +
				FemHigh.L2.NumbersByHIVstage[ia][0] + 
				FemHigh.L11.NumbersByHIVstage[ia][0] + FemHigh.L12.NumbersByHIVstage[ia][0] +
				FemHigh.L21.NumbersByHIVstage[ia][0] + FemHigh.L22.NumbersByHIVstage[ia][0] +
				FemLow.L1.NumbersByHIVstage[ia][0] + FemLow.L2.NumbersByHIVstage[ia][0];
		}*/
	}
	if(HIVind==1 && FixedUncertainty==1){
		OutNewHIV.out[CurrSim-1][CurrYear-StartYear] = 0.0;
		/*OutNewHIVMSM.out[CurrSim-1][CurrYear-StartYear] = 0.0;
		OutNewHIVMSF.out[CurrSim-1][CurrYear-StartYear] = 0.0;
		OutNewHIVMLM.out[CurrSim-1][CurrYear-StartYear] = 0.0;
		OutNewHIVMLF.out[CurrSim-1][CurrYear-StartYear] = 0.0;
		OutNewHIVUSM.out[CurrSim-1][CurrYear-StartYear] = 0.0;
		OutNewHIVUSF.out[CurrSim-1][CurrYear-StartYear] = 0.0;
		OutNewHIVCSM.out[CurrSim-1][CurrYear-StartYear] = 0.0;
		OutNewHIVCSF.out[CurrSim-1][CurrYear-StartYear] = 0.0;*/
		/*OutHIVinc15to24M.out[CurrSim-1][CurrYear-StartYear] = (NewHIVsum[1][0] +
			NewHIVsum[2][0])/(NegAtStart[1][0] + NegAtStart[2][0]);
		OutHIVinc15to24F.out[CurrSim-1][CurrYear-StartYear] = (NewHIVsum[1][1] +
			NewHIVsum[2][1])/(NegAtStart[1][1] + NegAtStart[2][1]);
		OutHIVinc25to49M.out[CurrSim-1][CurrYear-StartYear] = (NewHIVsum[3][0] + NewHIVsum[4][0] + 
			NewHIVsum[5][0] + NewHIVsum[6][0] + NewHIVsum[7][0])/(NegAtStart[3][0] +
			NegAtStart[4][0] + NegAtStart[5][0] + NegAtStart[6][0] + NegAtStart[7][0]);
		OutHIVinc25to49F.out[CurrSim-1][CurrYear-StartYear] = (NewHIVsum[3][1] + NewHIVsum[4][1] + 
			NewHIVsum[5][1] + NewHIVsum[6][1] + NewHIVsum[7][1])/(NegAtStart[3][1] +
			NegAtStart[4][1] + NegAtStart[5][1] + NegAtStart[6][1] + NegAtStart[7][1]);*/
		temp1 = 0.0;
		temp2 = 0.0;
		for(ia=1; ia<=7; ia++){
			temp1 += NewHIVsum[ia][0] + NewHIVsum[ia][1];
			temp2 += NegAtStart[ia][0] + NegAtStart[ia][1];
		}
		OutHIVinc15to49.out[CurrSim-1][CurrYear-StartYear] = temp1/temp2;
		if(CurrYear==2007){
			for(ia=1; ia<=9; ia++){
				OutHIVincByAgeM.out[CurrSim-1][ia-1] = NewHIVsum[ia][0]/NegAtStart[ia][0];
				OutHIVincByAgeF.out[CurrSim-1][ia-1] = NewHIVsum[ia][1]/NegAtStart[ia][1];
			}
		}
		for(ia=0; ia<16; ia++){
			OutNewHIV.out[CurrSim-1][CurrYear-StartYear] += NewHIVsum[ia][0] + 
				NewHIVsum[ia][1];
			/*OutNewHIVMSM.out[CurrSim-1][CurrYear-StartYear] += NewHIVsumMS[ia][0];
			OutNewHIVMSF.out[CurrSim-1][CurrYear-StartYear] += NewHIVsumMS[ia][1];
			OutNewHIVMLM.out[CurrSim-1][CurrYear-StartYear] += NewHIVsumML[ia][0];
			OutNewHIVMLF.out[CurrSim-1][CurrYear-StartYear] += NewHIVsumML[ia][1];
			OutNewHIVUSM.out[CurrSim-1][CurrYear-StartYear] += NewHIVsumUS[ia][0];
			OutNewHIVUSF.out[CurrSim-1][CurrYear-StartYear] += NewHIVsumUS[ia][1];
			OutNewHIVCSM.out[CurrSim-1][CurrYear-StartYear] += NewHIVsumCS[ia][0];
			OutNewHIVCSF.out[CurrSim-1][CurrYear-StartYear] += NewHIVsumCS[ia][1];*/
			NewHIVsum[ia][0] = 0.0;
			NewHIVsum[ia][1] = 0.0;
			/*NewHIVsumMS[ia][0] = 0.0;
			NewHIVsumMS[ia][1] = 0.0;
			NewHIVsumML[ia][0] = 0.0;
			NewHIVsumML[ia][1] = 0.0;
			NewHIVsumUS[ia][0] = 0.0;
			NewHIVsumUS[ia][1] = 0.0;
			NewHIVsumCS[ia][0] = 0.0;
			NewHIVsumCS[ia][1] = 0.0;*/
			/*if(CurrYear==2007){
				OutNewHIVMM.out[CurrSim-1][ia] = NewHIVsumM[ia][0];
				OutNewHIVUM.out[CurrSim-1][ia] = NewHIVsumU[ia][0];
				OutNewHIVMF.out[CurrSim-1][ia] = NewHIVsumM[ia][1];
				OutNewHIVUF.out[CurrSim-1][ia] = NewHIVsumU[ia][1];
				NewHIVsumM[ia][0] = 0.0;
				NewHIVsumM[ia][1] = 0.0;
				NewHIVsumU[ia][0] = 0.0;
				NewHIVsumU[ia][1] = 0.0;
			}*/
		}
		OutNewHIV.out[CurrSim-1][CurrYear-StartYear] += NewHIVperinatal + NewHIVbreastmilk;
		/*OutAIDSdeaths.out[CurrSim-1][CurrYear-StartYear] = 0.0;
		for(ia=0; ia<16; ia++){
			OutAIDSdeaths.out[CurrSim-1][CurrYear-StartYear] += NewAIDSdeaths[ia][0] + 
				NewAIDSdeaths[ia][1];
			NewAIDSdeaths[ia][0] = 0.0;
			NewAIDSdeaths[ia][1] = 0.0;
		}
		for(ia=0; ia<15; ia++){
			OutAIDSdeaths.out[CurrSim-1][CurrYear-StartYear] += MaleChild.AIDSdeathsTot[ia] + 
				FemChild.AIDSdeathsTot[ia];}*/
		// Calculate coefficient of variantion in HIV incidence rates
		//if(CurrYear==1999){
			temp1 = 0.0;
			temp2 = 0.0;
			for(ia=1; ia<8; ia++){
				temp1 += MaleHigh.L1.NumbersByHIVstage[ia][0] * MaleHigh.L1.HIVinfectProb[ia]; 
				temp1 += MaleHigh.L2.NumbersByHIVstage[ia][0] * MaleHigh.L2.HIVinfectProb[ia];
				temp1 += MaleHigh.L11.NumbersByHIVstage[ia][0] * MaleHigh.L11.HIVinfectProb[ia];
				temp1 += MaleHigh.L12.NumbersByHIVstage[ia][0] * MaleHigh.L12.HIVinfectProb[ia];
				temp1 += MaleHigh.L21.NumbersByHIVstage[ia][0] * MaleHigh.L21.HIVinfectProb[ia];
				temp1 += MaleHigh.L22.NumbersByHIVstage[ia][0] * MaleHigh.L22.HIVinfectProb[ia];
				temp1 += MaleHigh.S1.NumbersByHIVstage[ia][0] * MaleHigh.S1.HIVinfectProb[ia];
				temp1 += MaleHigh.S2.NumbersByHIVstage[ia][0] * MaleHigh.S2.HIVinfectProb[ia];
				temp1 += MaleHigh.S11.NumbersByHIVstage[ia][0] * MaleHigh.S11.HIVinfectProb[ia];
				temp1 += MaleHigh.S12.NumbersByHIVstage[ia][0] * MaleHigh.S12.HIVinfectProb[ia];
				temp1 += MaleHigh.S22.NumbersByHIVstage[ia][0] * MaleHigh.S22.HIVinfectProb[ia];
				temp1 += MaleHigh.NoPartner.NumbersByHIVstage[ia][0] * MaleHigh.NoPartner.HIVinfectProb[ia];
				temp1 += MaleLow.L1.NumbersByHIVstage[ia][0] * MaleLow.L1.HIVinfectProb[ia]; 
				temp1 += MaleLow.L2.NumbersByHIVstage[ia][0] * MaleLow.L2.HIVinfectProb[ia];
				temp1 += MaleLow.S1.NumbersByHIVstage[ia][0] * MaleLow.S1.HIVinfectProb[ia];
				temp1 += MaleLow.S2.NumbersByHIVstage[ia][0] * MaleLow.S2.HIVinfectProb[ia];
				temp1 += MaleLow.NoPartner.NumbersByHIVstage[ia][0] * MaleLow.NoPartner.HIVinfectProb[ia];
				temp1 += FemHigh.L1.NumbersByHIVstage[ia][0] * FemHigh.L1.HIVinfectProb[ia]; 
				temp1 += FemHigh.L2.NumbersByHIVstage[ia][0] * FemHigh.L2.HIVinfectProb[ia];
				temp1 += FemHigh.L11.NumbersByHIVstage[ia][0] * FemHigh.L11.HIVinfectProb[ia];
				temp1 += FemHigh.L12.NumbersByHIVstage[ia][0] * FemHigh.L12.HIVinfectProb[ia];
				temp1 += FemHigh.L21.NumbersByHIVstage[ia][0] * FemHigh.L21.HIVinfectProb[ia];
				temp1 += FemHigh.L22.NumbersByHIVstage[ia][0] * FemHigh.L22.HIVinfectProb[ia];
				temp1 += FemHigh.S1.NumbersByHIVstage[ia][0] * FemHigh.S1.HIVinfectProb[ia];
				temp1 += FemHigh.S2.NumbersByHIVstage[ia][0] * FemHigh.S2.HIVinfectProb[ia];
				temp1 += FemHigh.S11.NumbersByHIVstage[ia][0] * FemHigh.S11.HIVinfectProb[ia];
				temp1 += FemHigh.S12.NumbersByHIVstage[ia][0] * FemHigh.S12.HIVinfectProb[ia];
				temp1 += FemHigh.S22.NumbersByHIVstage[ia][0] * FemHigh.S22.HIVinfectProb[ia];
				temp1 += FemHigh.FSW.NumbersByHIVstage[ia][0] * FemHigh.S22.HIVinfectProb[ia];
				temp1 += FemHigh.NoPartner.NumbersByHIVstage[ia][0] * FemHigh.NoPartner.HIVinfectProb[ia];
				temp1 += FemLow.L1.NumbersByHIVstage[ia][0] * FemLow.L1.HIVinfectProb[ia]; 
				temp1 += FemLow.L2.NumbersByHIVstage[ia][0] * FemLow.L2.HIVinfectProb[ia];
				temp1 += FemLow.S1.NumbersByHIVstage[ia][0] * FemLow.S1.HIVinfectProb[ia];
				temp1 += FemLow.S2.NumbersByHIVstage[ia][0] * FemLow.S2.HIVinfectProb[ia];
				temp1 += FemLow.NoPartner.NumbersByHIVstage[ia][0] * FemLow.NoPartner.HIVinfectProb[ia];
				temp2 += MaleHigh.L1.NumbersByHIVstage[ia][0]; 
				temp2 += MaleHigh.L2.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.L11.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.L12.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.L21.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.L22.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.S1.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.S2.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.S11.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.S12.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.S22.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.NoPartner.NumbersByHIVstage[ia][0];
				temp2 += MaleHigh.Virgin.NumbersByHIVstage[ia][0];
				temp2 += MaleLow.L1.NumbersByHIVstage[ia][0]; 
				temp2 += MaleLow.L2.NumbersByHIVstage[ia][0];
				temp2 += MaleLow.S1.NumbersByHIVstage[ia][0];
				temp2 += MaleLow.S2.NumbersByHIVstage[ia][0];
				temp2 += MaleLow.NoPartner.NumbersByHIVstage[ia][0];
				temp2 += MaleLow.Virgin.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.L1.NumbersByHIVstage[ia][0]; 
				temp2 += FemHigh.L2.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.L11.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.L12.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.L21.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.L22.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.S1.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.S2.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.S11.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.S12.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.S22.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.FSW.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.NoPartner.NumbersByHIVstage[ia][0];
				temp2 += FemHigh.Virgin.NumbersByHIVstage[ia][0];
				temp2 += FemLow.L1.NumbersByHIVstage[ia][0]; 
				temp2 += FemLow.L2.NumbersByHIVstage[ia][0];
				temp2 += FemLow.S1.NumbersByHIVstage[ia][0];
				temp2 += FemLow.S2.NumbersByHIVstage[ia][0];
				temp2 += FemLow.NoPartner.NumbersByHIVstage[ia][0];
				temp2 += FemLow.Virgin.NumbersByHIVstage[ia][0];
			}
			mean = temp1/temp2;
			var = 0.0;
			for(ia=1; ia<8; ia++){
				var += MaleHigh.L1.NumbersByHIVstage[ia][0] * pow(MaleHigh.L1.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.L2.NumbersByHIVstage[ia][0] * pow(MaleHigh.L2.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.L11.NumbersByHIVstage[ia][0] * pow(MaleHigh.L11.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.L12.NumbersByHIVstage[ia][0] * pow(MaleHigh.L12.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.L21.NumbersByHIVstage[ia][0] * pow(MaleHigh.L21.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.L22.NumbersByHIVstage[ia][0] * pow(MaleHigh.L22.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.S1.NumbersByHIVstage[ia][0] * pow(MaleHigh.S1.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.S2.NumbersByHIVstage[ia][0] * pow(MaleHigh.S2.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.S11.NumbersByHIVstage[ia][0] * pow(MaleHigh.S11.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.S12.NumbersByHIVstage[ia][0] * pow(MaleHigh.S12.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.S22.NumbersByHIVstage[ia][0] * pow(MaleHigh.S22.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.NoPartner.NumbersByHIVstage[ia][0] * pow(MaleHigh.NoPartner.HIVinfectProb[ia] - mean, 2.0);
				var += MaleHigh.Virgin.NumbersByHIVstage[ia][0] * pow(mean, 2.0);
				var += MaleLow.L1.NumbersByHIVstage[ia][0] * pow(MaleLow.L1.HIVinfectProb[ia] - mean, 2.0);
				var += MaleLow.L2.NumbersByHIVstage[ia][0] * pow(MaleLow.L2.HIVinfectProb[ia] - mean, 2.0);
				var += MaleLow.S1.NumbersByHIVstage[ia][0] * pow(MaleLow.S1.HIVinfectProb[ia] - mean, 2.0);
				var += MaleLow.S2.NumbersByHIVstage[ia][0] * pow(MaleLow.S2.HIVinfectProb[ia] - mean, 2.0);
				var += MaleLow.NoPartner.NumbersByHIVstage[ia][0] * pow(MaleLow.NoPartner.HIVinfectProb[ia] - mean, 2.0);
				var += MaleLow.Virgin.NumbersByHIVstage[ia][0] * pow(mean, 2.0);
				var += FemHigh.L1.NumbersByHIVstage[ia][0] * pow(FemHigh.L1.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.L2.NumbersByHIVstage[ia][0] * pow(FemHigh.L2.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.L11.NumbersByHIVstage[ia][0] * pow(FemHigh.L11.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.L12.NumbersByHIVstage[ia][0] * pow(FemHigh.L12.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.L21.NumbersByHIVstage[ia][0] * pow(FemHigh.L21.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.L22.NumbersByHIVstage[ia][0] * pow(FemHigh.L22.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.S1.NumbersByHIVstage[ia][0] * pow(FemHigh.S1.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.S2.NumbersByHIVstage[ia][0] * pow(FemHigh.S2.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.S11.NumbersByHIVstage[ia][0] * pow(FemHigh.S11.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.S12.NumbersByHIVstage[ia][0] * pow(FemHigh.S12.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.S22.NumbersByHIVstage[ia][0] * pow(FemHigh.S22.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.FSW.NumbersByHIVstage[ia][0] * pow(FemHigh.FSW.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.NoPartner.NumbersByHIVstage[ia][0] * pow(FemHigh.NoPartner.HIVinfectProb[ia] - mean, 2.0);
				var += FemHigh.Virgin.NumbersByHIVstage[ia][0] * pow(mean, 2.0);
				var += FemLow.L1.NumbersByHIVstage[ia][0] * pow(FemLow.L1.HIVinfectProb[ia] - mean, 2.0);
				var += FemLow.L2.NumbersByHIVstage[ia][0] * pow(FemLow.L2.HIVinfectProb[ia] - mean, 2.0);
				var += FemLow.S1.NumbersByHIVstage[ia][0] * pow(FemLow.S1.HIVinfectProb[ia] - mean, 2.0);
				var += FemLow.S2.NumbersByHIVstage[ia][0] * pow(FemLow.S2.HIVinfectProb[ia] - mean, 2.0);
				var += FemLow.NoPartner.NumbersByHIVstage[ia][0] * pow(FemLow.NoPartner.HIVinfectProb[ia] - mean, 2.0);
				var += FemLow.Virgin.NumbersByHIVstage[ia][0] * pow(mean, 2.0);
			}
			OutCoVHIVinc.out[CurrSim-1][CurrYear-StartYear] = pow(var/temp2, 0.5)/mean;
		//}
	}

	if(HSVind==1){
		HSVtransitionM.ClearAlive();
		HSVtransitionF.ClearAlive();
	}
	if(TPind==1){
		TPtransitionM.ClearAlive();
		TPtransitionF.ClearAlive();
	}
	if(HDind==1){
		HDtransitionM.ClearAlive();
		HDtransitionF.ClearAlive();
	}
	if(NGind==1){
		NGtransitionM.ClearAlive();
		NGtransitionF.ClearAlive();
	}
	if(CTind==1){
		CTtransitionM.ClearAlive();
		CTtransitionF.ClearAlive();
	}
	if(TVind==1){
		TVtransitionM.ClearAlive();
		TVtransitionF.ClearAlive();
	}
	if(BVind==1){BVtransitionF.ClearAlive();}
	if(VCind==1){VCtransitionF.ClearAlive();}

	MaleHigh.GetTotalBySTDstage();
	MaleLow.GetTotalBySTDstage();
	FemHigh.GetTotalBySTDstage();
	FemLow.GetTotalBySTDstage();

	if(HSVind==1){
		HSVtransitionM.CalcTotalAlive();
		HSVtransitionF.CalcTotalAlive();
	}
	if(TPind==1){
		TPtransitionM.CalcTotalAlive();
		TPtransitionF.CalcTotalAlive();
	}
	if(HDind==1){
		HDtransitionM.CalcTotalAlive();
		HDtransitionF.CalcTotalAlive();
	}
	if(NGind==1){
		NGtransitionM.CalcTotalAlive();
		NGtransitionF.CalcTotalAlive();
	}
	if(CTind==1){
		CTtransitionM.CalcTotalAlive();
		CTtransitionF.CalcTotalAlive();
	}
	if(TVind==1){
		TVtransitionM.CalcTotalAlive();
		TVtransitionF.CalcTotalAlive();
	}
	if(BVind==1){BVtransitionF.CalcTotalAlive();}
	if(VCind==1){VCtransitionF.CalcTotalAlive();}
}

void OneBehavCycle()
{
	int ic, STDcyclesPerBehavCycle;

	// Call the OneSTDcycle the appropriate number of times
	STDcyclesPerBehavCycle = CycleD/CycleS;
	for(ic=0; ic<STDcyclesPerBehavCycle; ic++){
		OneSTDcycle();}
	
	// Calculate movements between sexual behaviour classes
	UpdateMarriageIncidence();
	BalanceSexualPartners();
	GetPartnerAIDSmort();
	CalcPartnerTransitions();

	// Update the numbers in each HIV stage and propns in each STD stage, for each sexual
	// behaviour group
	MaleHigh.SetHIVnumbersToTemp();
	MaleLow.SetHIVnumbersToTemp();
	FemHigh.SetHIVnumbersToTemp();
	FemLow.SetHIVnumbersToTemp();
	MaleHigh.Virgin.SetHIVnumbersToTemp();
	MaleLow.Virgin.SetHIVnumbersToTemp();
	FemHigh.Virgin.SetHIVnumbersToTemp();
	FemLow.Virgin.SetHIVnumbersToTemp();
	MaleHigh.GetAllPropnsBySTDstage();
	MaleLow.GetAllPropnsBySTDstage();
	FemHigh.GetAllPropnsBySTDstage();
	FemLow.GetAllPropnsBySTDstage();
}

void UpdateMarriageIncidence()
{
	int ia, ii, ij;
	double CurrentSTpartners[2][2]; // Current ST partnerships, by male risk group (1st index)
									// and female risk group (second index)
	double ActualPropnSTH[2][2]; // Proportion of ST partners who are in the high-risk
								 // group, by risk group (1st index) and sex (2nd index)

	// First determine the MarriageRate adjustment factors
	for(ii=0; ii<2; ii++){
		for(ij=0; ij<2; ij++){
			CurrentSTpartners[ii][ij] = 0;}
	}
	for(ia=0; ia<16; ia++){
		CurrentSTpartners[0][0] += MaleHigh.S1.TotalAlive[ia] + MaleHigh.S11.TotalAlive[ia] *
			2.0 + MaleHigh.S12.TotalAlive[ia] + MaleHigh.L11.TotalAlive[ia] + 
			MaleHigh.L21.TotalAlive[ia];
		CurrentSTpartners[0][1] += MaleHigh.S2.TotalAlive[ia] + MaleHigh.S22.TotalAlive[ia] *
			2.0 + MaleHigh.S12.TotalAlive[ia] + MaleHigh.L12.TotalAlive[ia] + 
			MaleHigh.L22.TotalAlive[ia];
		CurrentSTpartners[1][0] += MaleLow.S1.TotalAlive[ia];
		CurrentSTpartners[1][1] += MaleLow.S2.TotalAlive[ia];
	}
	ActualPropnSTH[0][0] = CurrentSTpartners[0][0]/(CurrentSTpartners[0][0] + 
		CurrentSTpartners[0][1]);
	ActualPropnSTH[1][0] = CurrentSTpartners[1][0]/(CurrentSTpartners[1][0] + 
		CurrentSTpartners[1][1]);
	ActualPropnSTH[0][1] = CurrentSTpartners[0][0]/(CurrentSTpartners[0][0] + 
		CurrentSTpartners[1][0]);
	ActualPropnSTH[1][1] = CurrentSTpartners[0][1]/(CurrentSTpartners[0][1] + 
		CurrentSTpartners[1][1]);
	MarriageRate[0][0] = ((1.0 - ActualPropnSTH[0][0])/ActualPropnSTH[0][0])/
		((1.0 - ActualPropnLTH[0][0])/ActualPropnLTH[0][0]);
	MarriageRate[0][1] = ((1.0 - ActualPropnSTH[1][0])/ActualPropnSTH[1][0])/
		((1.0 - ActualPropnLTH[1][0])/ActualPropnLTH[1][0]);
	MarriageRate[0][2] = ((1.0 - ActualPropnSTH[0][1])/ActualPropnSTH[0][1])/
		((1.0 - ActualPropnLTH[0][1])/ActualPropnLTH[0][1]);
	MarriageRate[0][3] = ((1.0 - ActualPropnSTH[1][1])/ActualPropnSTH[1][1])/
		((1.0 - ActualPropnLTH[1][1])/ActualPropnLTH[1][1]);
	for(ij=0; ij<4; ij++){
		MarriageRate[1][ij] = 1.0;}

	// Then calculate AgeEffectMarriage
	for(ia=0; ia<16; ia++){
		AgeEffectMarriage[ia][0] = MarriageIncidence[ia][0] * (MaleHigh.Virgin.TotalAlive[ia] +
			MaleHigh.NoPartner.TotalAlive[ia] + MaleHigh.S1.TotalAlive[ia] + 
			MaleHigh.S2.TotalAlive[ia] + MaleHigh.S11.TotalAlive[ia] + 
			MaleHigh.S12.TotalAlive[ia] + MaleHigh.S22.TotalAlive[ia])/
			((MaleHigh.S1.TotalAlive[ia] + MaleHigh.S11.TotalAlive[ia]*2.0) * MarriageRate[0][0] + 
			(MaleHigh.S2.TotalAlive[ia] + MaleHigh.S22.TotalAlive[ia]*2.0) * MarriageRate[1][0] +
			MaleHigh.S12.TotalAlive[ia] * (MarriageRate[0][0] + MarriageRate[1][0]));
		AgeEffectMarriage[ia][1] = MarriageIncidence[ia][0] * (MaleLow.Virgin.TotalAlive[ia] +
			MaleLow.NoPartner.TotalAlive[ia] + MaleLow.S1.TotalAlive[ia] + 
			MaleLow.S2.TotalAlive[ia])/(MaleLow.S1.TotalAlive[ia] * MarriageRate[0][1] + 
			MaleLow.S2.TotalAlive[ia] * MarriageRate[1][1]);
		AgeEffectMarriage[ia][2] = MarriageIncidence[ia][1] * (FemHigh.Virgin.TotalAlive[ia] +
			FemHigh.NoPartner.TotalAlive[ia] + FemHigh.FSW.TotalAlive[ia] + 
			FemHigh.S1.TotalAlive[ia] + FemHigh.S2.TotalAlive[ia] + FemHigh.S11.TotalAlive[ia] + 
			FemHigh.S12.TotalAlive[ia] + FemHigh.S22.TotalAlive[ia])/
			((FemHigh.S1.TotalAlive[ia] + FemHigh.S11.TotalAlive[ia]*2.0) * MarriageRate[0][2] + 
			(FemHigh.S2.TotalAlive[ia] + FemHigh.S22.TotalAlive[ia]*2.0) * MarriageRate[1][2] +
			FemHigh.S12.TotalAlive[ia] * (MarriageRate[0][2] + MarriageRate[1][2]));
		AgeEffectMarriage[ia][3] = MarriageIncidence[ia][1] * (FemLow.Virgin.TotalAlive[ia] +
			FemLow.NoPartner.TotalAlive[ia] + FemLow.S1.TotalAlive[ia] + 
			FemLow.S2.TotalAlive[ia])/(FemLow.S1.TotalAlive[ia] * MarriageRate[0][3] + 
			FemLow.S2.TotalAlive[ia] * MarriageRate[1][3]);
	}
}

void BalanceSexualPartners()
{
	// This function performs the calculations in the "Calcs" sheet in the Excel model.

	int ia, ii, ij;
	double TotalFSW;

	MaleHigh.GetNewPartners();
	MaleLow.GetNewPartners();
	FemHigh.GetNewPartners();
	FemLow.GetNewPartners();

	DesiredSTpartners[0][0] = MaleHigh.NoPartner.DesiredNewPartners + 
		MaleHigh.S1.DesiredNewPartners + MaleHigh.S2.DesiredNewPartners + 
		MaleHigh.L1.DesiredNewPartners + MaleHigh.L2.DesiredNewPartners;
	DesiredSTpartners[0][1] = FemHigh.NoPartner.DesiredNewPartners + 
		FemHigh.S1.DesiredNewPartners + FemHigh.S2.DesiredNewPartners + 
		FemHigh.L1.DesiredNewPartners + FemHigh.L2.DesiredNewPartners;
	DesiredSTpartners[1][0] = MaleLow.NoPartner.DesiredNewPartners;
	DesiredSTpartners[1][1] = FemLow.NoPartner.DesiredNewPartners;

	DesiredPartnerRiskM[0][0] = (1.0 - AssortativeM) + AssortativeM * 
		DesiredSTpartners[0][1] / (DesiredSTpartners[0][1] + DesiredSTpartners[1][1]);
	DesiredPartnerRiskM[0][1] = 1.0 - DesiredPartnerRiskM[0][0];
	DesiredPartnerRiskM[1][1] = (1.0 - AssortativeM) + AssortativeM * 
		DesiredSTpartners[1][1] / (DesiredSTpartners[0][1] + DesiredSTpartners[1][1]);
	DesiredPartnerRiskM[1][0] = 1.0 - DesiredPartnerRiskM[1][1];

	DesiredPartnerRiskF[0][0] = (1.0 - AssortativeF) + AssortativeF * 
		DesiredSTpartners[0][0] / (DesiredSTpartners[0][0] + DesiredSTpartners[1][0]);
	DesiredPartnerRiskF[0][1] = 1.0 - DesiredPartnerRiskF[0][0];
	DesiredPartnerRiskF[1][1] = (1.0 - AssortativeF) + AssortativeF * 
		DesiredSTpartners[1][0] / (DesiredSTpartners[0][0] + DesiredSTpartners[1][0]);
	DesiredPartnerRiskF[1][0] = 1.0 - DesiredPartnerRiskF[1][1];

	for(ii=0; ii<2; ii++){
		for(ij=0; ij<2; ij++){
			AdjSTrateM[ii][ij] = (GenderEquality * DesiredSTpartners[ij][1] * 
				DesiredPartnerRiskF[ij][ii] + (1.0 - GenderEquality) * 
				DesiredSTpartners[ii][0] * DesiredPartnerRiskM[ii][ij])/
				(DesiredSTpartners[ii][0] * DesiredPartnerRiskM[ii][ij]);
			AdjSTrateF[ii][ij] = (GenderEquality * DesiredSTpartners[ii][1] * 
				DesiredPartnerRiskF[ii][ij] + (1.0 - GenderEquality) * 
				DesiredSTpartners[ij][0] * DesiredPartnerRiskM[ij][ii])/
				(DesiredSTpartners[ii][1] * DesiredPartnerRiskF[ii][ij]);
		}
	}

	// Check that rate of extramarital contact in females is < that in males
	if(PartnerEffectNew[1][1]*(DesiredPartnerRiskF[0][0] * AdjSTrateF[0][0] + 
		DesiredPartnerRiskF[0][1] * AdjSTrateF[0][1]) > PartnerEffectNew[1][0]*
		(DesiredPartnerRiskM[0][0] * AdjSTrateM[0][0] + DesiredPartnerRiskM[0][1] * 
		AdjSTrateM[0][1])){
			ErrorInd = 1;}

	DesiredMarriagesM[0][0] = MaleHigh.S1.DesiredNewL1 + MaleHigh.S11.DesiredNewL1 +
		MaleHigh.S12.DesiredNewL1;
	DesiredMarriagesM[0][1] = MaleHigh.S2.DesiredNewL2 + MaleHigh.S12.DesiredNewL2 +
		MaleHigh.S22.DesiredNewL2;
	DesiredMarriagesM[1][0] = MaleLow.S1.DesiredNewL1;
	DesiredMarriagesM[1][1] = MaleLow.S2.DesiredNewL2;

	DesiredMarriagesF[0][0] = FemHigh.S1.DesiredNewL1 + FemHigh.S11.DesiredNewL1 +
		FemHigh.S12.DesiredNewL1;
	DesiredMarriagesF[0][1] = FemHigh.S2.DesiredNewL2 + FemHigh.S12.DesiredNewL2 +
		FemHigh.S22.DesiredNewL2;
	DesiredMarriagesF[1][0] = FemLow.S1.DesiredNewL1;
	DesiredMarriagesF[1][1] = FemLow.S2.DesiredNewL2;

	for(ii=0; ii<2; ii++){
		for(ij=0; ij<2; ij++){
			AdjLTrateM[ii][ij] = (GenderEquality * DesiredMarriagesF[ij][ii] +
				(1.0 - GenderEquality) * DesiredMarriagesM[ii][ij])/DesiredMarriagesM[ii][ij];
			AdjLTrateF[ii][ij] = (GenderEquality * DesiredMarriagesF[ii][ij] + 
				(1.0 - GenderEquality) * DesiredMarriagesM[ij][ii])/DesiredMarriagesF[ii][ij];
		}
	}

	DesiredFSWcontacts = MaleHigh.NoPartner.AnnFSWcontacts + MaleHigh.S1.AnnFSWcontacts +
		MaleHigh.S2.AnnFSWcontacts + MaleHigh.S11.AnnFSWcontacts +
		MaleHigh.S12.AnnFSWcontacts + MaleHigh.S22.AnnFSWcontacts +
		MaleHigh.L1.AnnFSWcontacts + MaleHigh.L2.AnnFSWcontacts +
		MaleHigh.L11.AnnFSWcontacts + MaleHigh.L12.AnnFSWcontacts +
		MaleHigh.L21.AnnFSWcontacts + MaleHigh.L22.AnnFSWcontacts;
	TotalFSW = 0;
	for(ia=0; ia<16; ia++){
		TotalFSW += FemHigh.FSW.TotalAlive[ia];}
	RequiredNewFSW = DesiredFSWcontacts/AnnNumberClients - TotalFSW;
	//if(RequiredNewFSW<0.0){RequiredNewFSW = 0.0;}
}

void GetPartnerAIDSmort()
{
	int ia, id, iy;
	double denominator;

	for(ia=0; ia<16; ia++){
		// Calculate entries in AIDSmortForceM array
		denominator = MaleHigh.S1.TotalAlive[ia] +
			MaleHigh.S11.TotalAlive[ia] * 2.0 + MaleHigh.S12.TotalAlive[ia] +
			MaleHigh.L11.TotalAlive[ia] + MaleHigh.L21.TotalAlive[ia];
		if(denominator>0){
			AIDSmortForceM[ia][0] = -log(pow(1.0 - (MaleHigh.S1.HIVstageExits[ia][4] +
				MaleHigh.S1.HIVstageExits[ia][5] + (MaleHigh.S11.HIVstageExits[ia][4] +
				MaleHigh.S11.HIVstageExits[ia][5]) * 2.0 + MaleHigh.S12.HIVstageExits[ia][4] +
				MaleHigh.S12.HIVstageExits[ia][5] + MaleHigh.L11.HIVstageExits[ia][4] +
				MaleHigh.L11.HIVstageExits[ia][5] + MaleHigh.L21.HIVstageExits[ia][4] +
				MaleHigh.L21.HIVstageExits[ia][5])/denominator, CycleD));}
		else{AIDSmortForceM[ia][0] = 0;}
		denominator = MaleHigh.S2.TotalAlive[ia] +
			MaleHigh.S22.TotalAlive[ia] * 2.0 + MaleHigh.S12.TotalAlive[ia] +
			MaleHigh.L12.TotalAlive[ia] + MaleHigh.L22.TotalAlive[ia];
		if(denominator>0){
			AIDSmortForceM[ia][1] = -log(pow(1.0 - (MaleHigh.S2.HIVstageExits[ia][4] +
				MaleHigh.S2.HIVstageExits[ia][5] + (MaleHigh.S22.HIVstageExits[ia][4] +
				MaleHigh.S22.HIVstageExits[ia][5]) * 2.0 + MaleHigh.S12.HIVstageExits[ia][4] +
				MaleHigh.S12.HIVstageExits[ia][5] + MaleHigh.L12.HIVstageExits[ia][4] +
				MaleHigh.L12.HIVstageExits[ia][5] + MaleHigh.L22.HIVstageExits[ia][4] +
				MaleHigh.L22.HIVstageExits[ia][5])/denominator, CycleD));}
		else{AIDSmortForceM[ia][1] = 0;}
		if(MaleLow.S1.TotalAlive[ia]>0){
			AIDSmortForceM[ia][2] = -log(pow(1.0 - (MaleLow.S1.HIVstageExits[ia][4] +
				MaleLow.S1.HIVstageExits[ia][5])/MaleLow.S1.TotalAlive[ia], CycleD));}
		else{AIDSmortForceM[ia][2] = 0;}
		if(MaleLow.S2.TotalAlive[ia]>0){
			AIDSmortForceM[ia][3] = -log(pow(1.0 - (MaleLow.S2.HIVstageExits[ia][4] +
				MaleLow.S2.HIVstageExits[ia][5])/MaleLow.S2.TotalAlive[ia], CycleD));}
		else{AIDSmortForceM[ia][3] = 0;}
		denominator = MaleHigh.L1.TotalAlive[ia] + 
			MaleHigh.L11.TotalAlive[ia] + MaleHigh.L12.TotalAlive[ia];
		if(denominator>0){
			AIDSmortForceM[ia][4] = -log(pow(1.0 - (MaleHigh.L1.HIVstageExits[ia][4] + 
				MaleHigh.L1.HIVstageExits[ia][5] + MaleHigh.L11.HIVstageExits[ia][4] + 
				MaleHigh.L11.HIVstageExits[ia][5] + MaleHigh.L12.HIVstageExits[ia][4] + 
				MaleHigh.L12.HIVstageExits[ia][5])/denominator, CycleD));}
		else{AIDSmortForceM[ia][4] = 0;}
		denominator = MaleHigh.L2.TotalAlive[ia] + 
			MaleHigh.L21.TotalAlive[ia] + MaleHigh.L22.TotalAlive[ia];
		if(denominator>0){
			AIDSmortForceM[ia][5] = -log(pow(1.0 - (MaleHigh.L2.HIVstageExits[ia][4] + 
				MaleHigh.L2.HIVstageExits[ia][5] + MaleHigh.L21.HIVstageExits[ia][4] + 
				MaleHigh.L21.HIVstageExits[ia][5] + MaleHigh.L22.HIVstageExits[ia][4] + 
				MaleHigh.L22.HIVstageExits[ia][5])/denominator, CycleD));}
		else{AIDSmortForceM[ia][5] = 0;}
		if(MaleLow.L1.TotalAlive[ia]>0){
			AIDSmortForceM[ia][6] = -log(pow(1.0 - (MaleLow.L1.HIVstageExits[ia][4] +
				MaleLow.L1.HIVstageExits[ia][5])/MaleLow.L1.TotalAlive[ia], CycleD));}
		else{AIDSmortForceM[ia][6] = 0;}
		if(MaleLow.L2.TotalAlive[ia]>0){
			AIDSmortForceM[ia][7] = -log(pow(1.0 - (MaleLow.L2.HIVstageExits[ia][4] +
				MaleLow.L2.HIVstageExits[ia][5])/MaleLow.L2.TotalAlive[ia], CycleD));}
		else{AIDSmortForceM[ia][7] = 0;}

		// Calculate entries in AIDSmortForceF array
		denominator = FemHigh.S1.TotalAlive[ia] +
			FemHigh.S11.TotalAlive[ia] * 2.0 + FemHigh.S12.TotalAlive[ia] +
			FemHigh.L11.TotalAlive[ia] + FemHigh.L21.TotalAlive[ia];
		if(denominator>0){
			AIDSmortForceF[ia][0] = -log(pow(1.0 - (FemHigh.S1.HIVstageExits[ia][4] +
				FemHigh.S1.HIVstageExits[ia][5] + (FemHigh.S11.HIVstageExits[ia][4] +
				FemHigh.S11.HIVstageExits[ia][5]) * 2.0 + FemHigh.S12.HIVstageExits[ia][4] +
				FemHigh.S12.HIVstageExits[ia][5] + FemHigh.L11.HIVstageExits[ia][4] +
				FemHigh.L11.HIVstageExits[ia][5] + FemHigh.L21.HIVstageExits[ia][4] +
				FemHigh.L21.HIVstageExits[ia][5])/denominator, CycleD));}
		else{AIDSmortForceF[ia][0] = 0;}
		denominator = FemHigh.S2.TotalAlive[ia] +
			FemHigh.S22.TotalAlive[ia] * 2.0 + FemHigh.S12.TotalAlive[ia] +
			FemHigh.L12.TotalAlive[ia] + FemHigh.L22.TotalAlive[ia];
		if(denominator>0){
			AIDSmortForceF[ia][1] = -log(pow(1.0 - (FemHigh.S2.HIVstageExits[ia][4] +
				FemHigh.S2.HIVstageExits[ia][5] + (FemHigh.S22.HIVstageExits[ia][4] +
				FemHigh.S22.HIVstageExits[ia][5]) * 2.0 + FemHigh.S12.HIVstageExits[ia][4] +
				FemHigh.S12.HIVstageExits[ia][5] + FemHigh.L12.HIVstageExits[ia][4] +
				FemHigh.L12.HIVstageExits[ia][5] + FemHigh.L22.HIVstageExits[ia][4] +
				FemHigh.L22.HIVstageExits[ia][5])/denominator, CycleD));}
		else{AIDSmortForceF[ia][1] = 0;}
		if(FemLow.S1.TotalAlive[ia]>0){
			AIDSmortForceF[ia][2] = -log(pow(1.0 - (FemLow.S1.HIVstageExits[ia][4] +
				FemLow.S1.HIVstageExits[ia][5])/FemLow.S1.TotalAlive[ia], CycleD));}
		else{AIDSmortForceF[ia][2] = 0;}
		if(FemLow.S2.TotalAlive[ia]>0){
			AIDSmortForceF[ia][3] = -log(pow(1.0 - (FemLow.S2.HIVstageExits[ia][4] +
				FemLow.S2.HIVstageExits[ia][5])/FemLow.S2.TotalAlive[ia], CycleD));}
		else{AIDSmortForceF[ia][3] = 0;}
		denominator = FemHigh.L1.TotalAlive[ia] + 
			FemHigh.L11.TotalAlive[ia] + FemHigh.L12.TotalAlive[ia];
		if(denominator>0){
			AIDSmortForceF[ia][4] = -log(pow(1.0 - (FemHigh.L1.HIVstageExits[ia][4] + 
				FemHigh.L1.HIVstageExits[ia][5] + FemHigh.L11.HIVstageExits[ia][4] + 
				FemHigh.L11.HIVstageExits[ia][5] + FemHigh.L12.HIVstageExits[ia][4] + 
				FemHigh.L12.HIVstageExits[ia][5])/denominator, CycleD));}
		else{AIDSmortForceF[ia][4] = 0;}
		denominator = FemHigh.L2.TotalAlive[ia] + 
			FemHigh.L21.TotalAlive[ia] + FemHigh.L22.TotalAlive[ia];
		if(denominator>0){
			AIDSmortForceF[ia][5] = -log(pow(1.0 - (FemHigh.L2.HIVstageExits[ia][4] + 
				FemHigh.L2.HIVstageExits[ia][5] + FemHigh.L21.HIVstageExits[ia][4] + 
				FemHigh.L21.HIVstageExits[ia][5] + FemHigh.L22.HIVstageExits[ia][4] + 
				FemHigh.L22.HIVstageExits[ia][5])/denominator, CycleD));}
		else{AIDSmortForceF[ia][5] = 0;}
		if(FemLow.L1.TotalAlive[ia]>0){
			AIDSmortForceF[ia][6] = -log(pow(1.0 - (FemLow.L1.HIVstageExits[ia][4] +
				FemLow.L1.HIVstageExits[ia][5])/FemLow.L1.TotalAlive[ia], CycleD));}
		else{AIDSmortForceF[ia][6] = 0;}
		if(FemLow.L2.TotalAlive[ia]>0){
			AIDSmortForceF[ia][7] = -log(pow(1.0 - (FemLow.L2.HIVstageExits[ia][4] +
				FemLow.L2.HIVstageExits[ia][5])/FemLow.L2.TotalAlive[ia], CycleD));}
		else{AIDSmortForceF[ia][7] = 0;}
	}

	for(ia=0; ia<16; ia++){
		for(id=0; id<8; id++){
			// Calculate value in AIDSmortPartnerM array
			AIDSmortPartnerM[ia][id] = 0;
			for(iy=0; iy<16; iy++){
				AIDSmortPartnerM[ia][id] += AIDSmortForceF[iy][id] * AgePrefM[ia][iy];}
			// Calculate value in AIDSmortPartnerF array
			AIDSmortPartnerF[ia][id] = 0;
			for(iy=0; iy<16; iy++){
				AIDSmortPartnerF[ia][id] += AIDSmortForceM[iy][id] * AgePrefF[ia][iy];}
		}
	}
}

void CalcPartnerTransitions()
{
	MaleHigh.GetPartnerTransitions();
	MaleLow.GetPartnerTransitions();
	FemHigh.GetPartnerTransitions();
	FemLow.GetPartnerTransitions();

	// Get the dependent rates from the independent rates of transition
	MaleHigh.Virgin.ConvertToDependent2(MaleHigh.Virgin.AcquireNewS1, 
		MaleHigh.Virgin.AcquireNewS2, MaleHigh.Virgin.AcquireNewS1dep,
		MaleHigh.Virgin.AcquireNewS2dep);
	MaleHigh.NoPartner.ConvertToDependent2(MaleHigh.NoPartner.AcquireNewS1, 
		MaleHigh.NoPartner.AcquireNewS2, MaleHigh.NoPartner.AcquireNewS1dep,
		MaleHigh.NoPartner.AcquireNewS2dep);
	MaleHigh.S1.ConvertToDependent4(MaleHigh.S1.LoseS1, MaleHigh.S1.MarryL1, 
		MaleHigh.S1.AcquireNewS1, MaleHigh.S1.AcquireNewS2, MaleHigh.S1.LoseS1dep,
		MaleHigh.S1.MarryL1dep, MaleHigh.S1.AcquireNewS1dep, MaleHigh.S1.AcquireNewS2dep);
	MaleHigh.S2.ConvertToDependent4(MaleHigh.S2.LoseS2, MaleHigh.S2.MarryL2, 
		MaleHigh.S2.AcquireNewS1, MaleHigh.S2.AcquireNewS2, MaleHigh.S2.LoseS2dep,
		MaleHigh.S2.MarryL2dep, MaleHigh.S2.AcquireNewS1dep, MaleHigh.S2.AcquireNewS2dep);
	MaleHigh.L1.ConvertToDependent3(MaleHigh.L1.LoseL, MaleHigh.L1.AcquireNewS1, 
		MaleHigh.L1.AcquireNewS2, MaleHigh.L1.LoseLdep, MaleHigh.L1.AcquireNewS1dep, 
		MaleHigh.L1.AcquireNewS2dep);
	MaleHigh.L2.ConvertToDependent3(MaleHigh.L2.LoseL, MaleHigh.L2.AcquireNewS1, 
		MaleHigh.L2.AcquireNewS2, MaleHigh.L2.LoseLdep, MaleHigh.L2.AcquireNewS1dep, 
		MaleHigh.L2.AcquireNewS2dep);
	MaleHigh.S11.ConvertToDependent2(MaleHigh.S11.LoseS1, MaleHigh.S11.MarryL1, 
		MaleHigh.S11.LoseS1dep, MaleHigh.S11.MarryL1dep);
	MaleHigh.S12.ConvertToDependent4(MaleHigh.S12.LoseS1, MaleHigh.S12.LoseS2, 
		MaleHigh.S12.MarryL1, MaleHigh.S12.MarryL2, MaleHigh.S12.LoseS1dep, 
		MaleHigh.S12.LoseS2dep, MaleHigh.S12.MarryL1dep, MaleHigh.S12.MarryL2dep);
	MaleHigh.S22.ConvertToDependent2(MaleHigh.S22.LoseS2, MaleHigh.S22.MarryL2, 
		MaleHigh.S22.LoseS2dep, MaleHigh.S22.MarryL2dep);
	MaleHigh.L11.ConvertToDependent2(MaleHigh.L11.LoseS1, MaleHigh.L11.LoseL, 
		MaleHigh.L11.LoseS1dep, MaleHigh.L11.LoseLdep);
	MaleHigh.L12.ConvertToDependent2(MaleHigh.L12.LoseS2, MaleHigh.L12.LoseL, 
		MaleHigh.L12.LoseS2dep, MaleHigh.L12.LoseLdep);
	MaleHigh.L21.ConvertToDependent2(MaleHigh.L21.LoseS1, MaleHigh.L21.LoseL, 
		MaleHigh.L21.LoseS1dep, MaleHigh.L21.LoseLdep);
	MaleHigh.L22.ConvertToDependent2(MaleHigh.L22.LoseS2, MaleHigh.L22.LoseL, 
		MaleHigh.L22.LoseS2dep, MaleHigh.L22.LoseLdep);
	MaleLow.Virgin.ConvertToDependent2(MaleLow.Virgin.AcquireNewS1, 
		MaleLow.Virgin.AcquireNewS2, MaleLow.Virgin.AcquireNewS1dep,
		MaleLow.Virgin.AcquireNewS2dep);
	MaleLow.NoPartner.ConvertToDependent2(MaleLow.NoPartner.AcquireNewS1, 
		MaleLow.NoPartner.AcquireNewS2, MaleLow.NoPartner.AcquireNewS1dep,
		MaleLow.NoPartner.AcquireNewS2dep);
	MaleLow.S1.ConvertToDependent2(MaleLow.S1.LoseS1, MaleLow.S1.MarryL1, 
		MaleLow.S1.LoseS1dep, MaleLow.S1.MarryL1dep);
	MaleLow.S2.ConvertToDependent2(MaleLow.S2.LoseS2, MaleLow.S2.MarryL2, 
		MaleLow.S2.LoseS2dep, MaleLow.S2.MarryL2dep);
	MaleLow.L1.ConvertToDependent1(MaleLow.L1.LoseL, MaleLow.L1.LoseLdep);
	MaleLow.L2.ConvertToDependent1(MaleLow.L2.LoseL, MaleLow.L2.LoseLdep);

	FemHigh.Virgin.ConvertToDependent2(FemHigh.Virgin.AcquireNewS1, 
		FemHigh.Virgin.AcquireNewS2, FemHigh.Virgin.AcquireNewS1dep,
		FemHigh.Virgin.AcquireNewS2dep);
	FemHigh.FSW.ConvertToDependent1(FemHigh.FSW.LeaveSW, FemHigh.FSW.LeaveSWdep);
	FemHigh.NoPartner.ConvertToDependent3(FemHigh.NoPartner.EnterSW, 
		FemHigh.NoPartner.AcquireNewS1, FemHigh.NoPartner.AcquireNewS2, 
		FemHigh.NoPartner.EnterSWdep, FemHigh.NoPartner.AcquireNewS1dep, 
		FemHigh.NoPartner.AcquireNewS2dep);
	FemHigh.S1.ConvertToDependent4(FemHigh.S1.LoseS1, FemHigh.S1.MarryL1, 
		FemHigh.S1.AcquireNewS1, FemHigh.S1.AcquireNewS2, FemHigh.S1.LoseS1dep,
		FemHigh.S1.MarryL1dep, FemHigh.S1.AcquireNewS1dep, FemHigh.S1.AcquireNewS2dep);
	FemHigh.S2.ConvertToDependent4(FemHigh.S2.LoseS2, FemHigh.S2.MarryL2, 
		FemHigh.S2.AcquireNewS1, FemHigh.S2.AcquireNewS2, FemHigh.S2.LoseS2dep,
		FemHigh.S2.MarryL2dep, FemHigh.S2.AcquireNewS1dep, FemHigh.S2.AcquireNewS2dep);
	FemHigh.L1.ConvertToDependent3(FemHigh.L1.LoseL, FemHigh.L1.AcquireNewS1, 
		FemHigh.L1.AcquireNewS2, FemHigh.L1.LoseLdep, FemHigh.L1.AcquireNewS1dep, 
		FemHigh.L1.AcquireNewS2dep);
	FemHigh.L2.ConvertToDependent3(FemHigh.L2.LoseL, FemHigh.L2.AcquireNewS1, 
		FemHigh.L2.AcquireNewS2, FemHigh.L2.LoseLdep, FemHigh.L2.AcquireNewS1dep, 
		FemHigh.L2.AcquireNewS2dep);
	FemHigh.S11.ConvertToDependent2(FemHigh.S11.LoseS1, FemHigh.S11.MarryL1, 
		FemHigh.S11.LoseS1dep, FemHigh.S11.MarryL1dep);
	FemHigh.S12.ConvertToDependent4(FemHigh.S12.LoseS1, FemHigh.S12.LoseS2, 
		FemHigh.S12.MarryL1, FemHigh.S12.MarryL2, FemHigh.S12.LoseS1dep, 
		FemHigh.S12.LoseS2dep, FemHigh.S12.MarryL1dep, FemHigh.S12.MarryL2dep);
	FemHigh.S22.ConvertToDependent2(FemHigh.S22.LoseS2, FemHigh.S22.MarryL2, 
		FemHigh.S22.LoseS2dep, FemHigh.S22.MarryL2dep);
	FemHigh.L11.ConvertToDependent2(FemHigh.L11.LoseS1, FemHigh.L11.LoseL, 
		FemHigh.L11.LoseS1dep, FemHigh.L11.LoseLdep);
	FemHigh.L12.ConvertToDependent2(FemHigh.L12.LoseS2, FemHigh.L12.LoseL, 
		FemHigh.L12.LoseS2dep, FemHigh.L12.LoseLdep);
	FemHigh.L21.ConvertToDependent2(FemHigh.L21.LoseS1, FemHigh.L21.LoseL, 
		FemHigh.L21.LoseS1dep, FemHigh.L21.LoseLdep);
	FemHigh.L22.ConvertToDependent2(FemHigh.L22.LoseS2, FemHigh.L22.LoseL, 
		FemHigh.L22.LoseS2dep, FemHigh.L22.LoseLdep);
	FemLow.Virgin.ConvertToDependent2(FemLow.Virgin.AcquireNewS1, 
		FemLow.Virgin.AcquireNewS2, FemLow.Virgin.AcquireNewS1dep,
		FemLow.Virgin.AcquireNewS2dep);
	FemLow.NoPartner.ConvertToDependent2(FemLow.NoPartner.AcquireNewS1, 
		FemLow.NoPartner.AcquireNewS2, FemLow.NoPartner.AcquireNewS1dep,
		FemLow.NoPartner.AcquireNewS2dep);
	FemLow.S1.ConvertToDependent2(FemLow.S1.LoseS1, FemLow.S1.MarryL1, FemLow.S1.LoseS1dep, 
		FemLow.S1.MarryL1dep);
	FemLow.S2.ConvertToDependent2(FemLow.S2.LoseS2, FemLow.S2.MarryL2, FemLow.S2.LoseS2dep, 
		FemLow.S2.MarryL2dep);
	FemLow.L1.ConvertToDependent1(FemLow.L1.LoseL, FemLow.L1.LoseLdep);
	FemLow.L2.ConvertToDependent1(FemLow.L2.LoseL, FemLow.L2.LoseLdep);

	// Calculate the numbers remaining in each sexual behaviour group
	MaleHigh.GetAllNumbersRemaining();
	MaleLow.GetAllNumbersRemaining();
	FemHigh.GetAllNumbersRemaining();
	FemLow.GetAllNumbersRemaining();

	// Calculate the numbers after allowing for movements between sexual behaviour groups
	MaleHigh.GetAllNumbersChanging();
	MaleLow.GetAllNumbersChanging();
	FemHigh.GetAllNumbersChanging();
	FemLow.GetAllNumbersChanging();
}

void OneSTDcycle()
{
	GetAllNumbersBySTDstage();
	if(CofactorType==2){
		UpdateSyndromePropns();}
	GetAllSTDcofactors();
	GetNewInfections();
	CalcSTDtransitions();
}

void GetAllNumbersBySTDstage()
{
	MaleHigh.GetAllNumbersBySTDstage();
	MaleLow.GetAllNumbersBySTDstage();
	FemHigh.GetAllNumbersBySTDstage();
	FemLow.GetAllNumbersBySTDstage();
}

void UpdateSyndromePropns()
{
	MaleHigh.UpdateSyndromePropns();
	MaleLow.UpdateSyndromePropns();
	FemHigh.UpdateSyndromePropns();
	FemLow.UpdateSyndromePropns();
}

void GetAllSTDcofactors()
{
	MaleHigh.GetAllSTDcofactors();
	MaleLow.GetAllSTDcofactors();
	FemHigh.GetAllSTDcofactors();
	FemLow.GetAllSTDcofactors();
}

void SetMaxHIVprevToFSW()
{
	int ia;
	double numerator, denominator;

	numerator = 0; 
	denominator = 0;
	for(ia=0; ia<16; ia++){
		numerator += FemHigh.FSW.NumbersByHIVstage[ia][0];
		denominator += FemHigh.FSW.TotalAlive[ia];
	}
	MaxHIVprev = 1.0 - numerator/denominator;
}

void GetNewInfections()
{
	if(TPind==1){
		TPtransitionM.ClearTransmProb();
		TPtransitionF.ClearTransmProb();}
	if(HSVind==1){
		HSVtransitionM.ClearTransmProb();
		HSVtransitionF.ClearTransmProb();}
	if(HDind==1){
		HDtransitionM.ClearTransmProb();
		HDtransitionF.ClearTransmProb();}
	if(NGind==1){
		NGtransitionM.ClearTransmProb();
		NGtransitionF.ClearTransmProb();}
	if(CTind==1){
		CTtransitionM.ClearTransmProb();
		CTtransitionF.ClearTransmProb();}
	if(TVind==1){
		TVtransitionM.ClearTransmProb();
		TVtransitionF.ClearTransmProb();}
	if(HIVind==1){
		HIVtransitionM.ClearTransmProb();
		HIVtransitionF.ClearTransmProb();}

	MaleHigh.CalcTransmissionProb();
	MaleLow.CalcTransmissionProb();
	FemHigh.CalcTransmissionProb();
	FemLow.CalcTransmissionProb();

	if(TPind==1){
		TPtransitionM.CalcTransmProb();
		TPtransitionF.CalcTransmProb();
		TPtransitionM.CalcAllInfectProb();
		TPtransitionF.CalcAllInfectProb();
	}
	if(HSVind==1){
		HSVtransitionM.CalcTransmProb();
		HSVtransitionF.CalcTransmProb();
		HSVtransitionM.CalcAllInfectProb();
		HSVtransitionF.CalcAllInfectProb();
	}
	if(HDind==1){
		HDtransitionM.CalcTransmProb();
		HDtransitionF.CalcTransmProb();
		HDtransitionM.CalcAllInfectProb();
		HDtransitionF.CalcAllInfectProb();
	}
	if(NGind==1){
		NGtransitionM.CalcTransmProb();
		NGtransitionF.CalcTransmProb();
		NGtransitionM.CalcAllInfectProb();
		NGtransitionF.CalcAllInfectProb();
	}
	if(CTind==1){
		CTtransitionM.CalcTransmProb();
		CTtransitionF.CalcTransmProb();
		CTtransitionM.CalcAllInfectProb();
		CTtransitionF.CalcAllInfectProb();
	}
	if(TVind==1){
		TVtransitionM.CalcTransmProb();
		TVtransitionF.CalcTransmProb();
		TVtransitionM.CalcAllInfectProb();
		TVtransitionF.CalcAllInfectProb();
	}
	if(HIVind==1){
		HIVtransitionM.CalcTransmProb();
		HIVtransitionF.CalcTransmProb();
		HIVtransitionM.CalcAllInfectProb();
		HIVtransitionF.CalcAllInfectProb();
	}

	SetMaxHIVprevToFSW();
	MaleHigh.CalcInfectionProb();
	MaleLow.CalcInfectionProb();
	FemHigh.CalcInfectionProb();
	FemLow.CalcInfectionProb();
}

void CalcSTDtransitions()
{
	MaleHigh.CalcSTDtransitions();
	MaleLow.CalcSTDtransitions();
	FemHigh.CalcSTDtransitions();
	FemLow.CalcSTDtransitions();

	MaleHigh.HIVstageChanges();
	MaleLow.HIVstageChanges();
	FemHigh.HIVstageChanges();
	FemLow.HIVstageChanges();

	if(HIVind==1){
		MaleHigh.SetHIVnumbersToTemp();
		MaleLow.SetHIVnumbersToTemp();
		FemHigh.SetHIVnumbersToTemp();
		FemLow.SetHIVnumbersToTemp();
	}
}

void SimulateParameters()
{
	if(HIVcalib==1){SimulateHIVparameters();}
	if(SexCalib==1){SimulateSexParameters();}
	if(NGind==1){SimulateNGparameters();}
}

void SimulateHIVparameters()
{
	int ind, i, is;
	double x, y, a, b, p, q;
	double r[6]; // Random variables from U(0, 1)

	ind = 2;

	if(FixedUncertainty==0){
		int32 seed = time(0) + CurrSim + ErrorCount;
		TRandomMersenne rg(seed);
		for(i=0; i<HIVtransmProb.columns; i++){
			r[i] = rg.Random();
			RandomUniformHIV.out[CurrSim-1][i] = r[i];
		}
	}
	else{
		for(i=0; i<HIVtransmProb.columns; i++){
			r[i] = RandomUniformHIV.out[CurrSim-1][i];}
	}

	// Simulate M->F transmission prob in CSW-client encounters
	/*a = 8.97;
	b = 2981.0;
	p = r[0];
	q = 1 - r[0];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	HIVtransitionM.TransmProb[0] = x;
	HIVtransmProb.out[CurrSim-1][0] = x;*/

	// Simulate M->F transmission prob per sex act in short-term partnerships
	a = 5.6789;
	b = 467.56;
	p = r[0];
	q = 1 - r[0];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	InitHIVtransm[1][0] = x;
	HIVtransmProb.out[CurrSim-1][0] = x;

	// Simulate M->F transmission prob per sex act in spousal partnerships (low risk spouse)
	a = 3.99;
	b = 1991.01;
	p = r[1];
	q = 1 - r[1];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	InitHIVtransm[2][0] = x;
	HIVtransmProb.out[CurrSim-1][1] = x;

	// Simulate F->M transmission prob in CSW-client encounters
	/*a = 8.7;
	b = 281.3;
	p = r[3];
	q = 1 - r[3];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	HIVtransitionF.TransmProb[0] = x;
	HIVtransmProb.out[CurrSim-1][3] = x;*/

	// Simulate F->M transmission prob per sex act in short-term partnerships
	a = 10.99;
	b = 1088.01;
	p = r[2];
	q = 1 - r[2];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	InitHIVtransm[1][1] = x;
	HIVtransmProb.out[CurrSim-1][2] = x;

	// Simulate F->M transmission prob per sex act in spousal partnerships (low risk spouse)
	a = 3.99;
	b = 1991.01;
	p = r[3];
	q = 1 - r[3];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	InitHIVtransm[2][1] = x;
	HIVtransmProb.out[CurrSim-1][3] = x;

	// Set the current HIV transm probs to the initial values
	HIVtransitionM.TransmProb[0] = InitHIVtransm[0][0];
	HIVtransitionF.TransmProb[0] = InitHIVtransm[0][1];
	for(is=0; is<3; is++){
		HIVtransitionM.TransmProb[is+1] = InitHIVtransm[1][0];
		HIVtransitionM.TransmProb[is+4] = InitHIVtransm[2][0];
		HIVtransitionF.TransmProb[is+1] = InitHIVtransm[1][1];
		HIVtransitionF.TransmProb[is+4] = InitHIVtransm[2][1];
	}
	if(CofactorType==0){
		HIVtransitionM.TransmProb[1] *= 2.0;
		HIVtransitionM.TransmProb[3] *= 0.5;
		HIVtransitionM.TransmProb[4] *= 2.0;
		HIVtransitionM.TransmProb[6] *= 0.5;
		HIVtransitionF.TransmProb[1] *= 2.0;
		HIVtransitionF.TransmProb[3] *= 0.5;
		HIVtransitionF.TransmProb[4] *= 2.0;
		HIVtransitionF.TransmProb[6] *= 0.5;
	}

	// Simulate initial HIV prevalence in the high risk group
	InitHIVprevHigh = r[4] * 0.002;
	HIVtransmProb.out[CurrSim-1][4] = InitHIVprevHigh;

	// Simulate the relative infectiousness of individuals on ART
	a = 3.50;
	b = 31.50;
	p = r[5];
	q = 1 - r[5];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	HIVtransitionM.HIVinfecIncrease[4] = (HIVtransitionM.HIVinfecIncrease[3] + 1.0) * x - 1.0;
	HIVtransitionF.HIVinfecIncrease[4] = (HIVtransitionF.HIVinfecIncrease[3] + 1.0) * x - 1.0;
	HIVtransmProb.out[CurrSim-1][5] = x;
}

void SimulateSexParameters()
{
	int ind, i;
	double x, y, a, b, p, q;
	double r[8]; // Random variables from U(0, 1)

	ind = 2;

	if(FixedUncertainty==0){
		int32 seed = time(0) + CurrSim + ErrorCount + simulations;
		TRandomMersenne rg(seed);
		for(i=0; i<SexBehavParameters.columns; i++){
			r[i] = rg.Random();
			RandomUniformSex.out[CurrSim-1][i] = r[i];
		}
	}
	else{
		for(i=0; i<SexBehavParameters.columns; i++){
			r[i] = RandomUniformSex.out[CurrSim-1][i];}
	}

	// Simulate high risk proportion in males
	/*a = 30.4875;
	b = 37.2625;
	p = r[0];
	q = 1 - r[0];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	HighPropnM = x;
	SexBehavParameters.out[CurrSim-1][0] = x;*/

	// Simulate high risk proportion in females
	/*a = 21.7681;
	b = 40.4264;
	p = r[1];
	q = 1 - r[1];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	HighPropnF = x;
	SexBehavParameters.out[CurrSim-1][1] = x;*/

	// Simulate factor by which rate of partner acquisition is reduced when a high risk 
	// individual has one non-spousal partner
	PartnerEffectNew[0][0] = r[0];
	PartnerEffectNew[0][1] = r[1];
	SexBehavParameters.out[CurrSim-1][0] = r[0];
	SexBehavParameters.out[CurrSim-1][1] = r[1];

	// Simulate the assortativeness of sexual mixing
	a = 5.8;
	b = 3.867;
	p = r[2];
	q = 1 - r[2];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	AssortativeM = x;
	AssortativeF = x;
	SexBehavParameters.out[CurrSim-1][2] = x;

	// Simulate the mean duration of nonspousal relationships
	/*a = 9.0;
	b = 18.0;
	p = r[6];
	q = 1 - r[6];
	cdfgam(&ind,&p,&q,&x,&a,&b,0,0);
	MeanDurSTrel[0][0] = x;
	MeanDurSTrel[0][1] = x;
	MeanDurSTrel[1][0] = x;
	MeanDurSTrel[1][1] = x;
	SexBehavParameters.out[CurrSim-1][6] = x;*/

	// Simulate the mean gap between relationships (high risk males)
	/*a = 9.2517;
	b = 55.4939;
	p = r[7];
	q = 1 - r[7];
	cdfgam(&ind,&p,&q,&x,&a,&b,0,0);
	BasePartnerAcqH[0] = 1.0/x;
	SexBehavParameters.out[CurrSim-1][7] = x;*/

	// Simulate the mean gap between relationships (high risk females)
	/*a = 9.2517;
	b = 111.02;
	p = r[8];
	q = 1 - r[8];
	cdfgam(&ind,&p,&q,&x,&a,&b,0,0);
	BasePartnerAcqH[1] = 1.0/x;
	SexBehavParameters.out[CurrSim-1][8] = x;*/

	// Simulate factor by which rate of partner acquisition is reduced when a high risk 
	// individual has a spousal partner
	PartnerEffectNew[1][0] = r[3];
	PartnerEffectNew[1][1] = r[4];
	SexBehavParameters.out[CurrSim-1][3] = r[3];
	SexBehavParameters.out[CurrSim-1][4] = r[4];

	// Simulate the ratio of the rate of partner acquisition in low risk group to that in
	// high risk group (comparing single individuals of same sex and age)
	PartnershipFormation[1][0] = r[5];
	PartnershipFormation[1][1] = r[6];
	SexBehavParameters.out[CurrSim-1][5] = r[5];
	SexBehavParameters.out[CurrSim-1][6] = r[6];
	
	// Simulate the bias in female reporting of condom use
	// NB. This is not the prior distribution but the distribution used to sample the starting points.
	a = 28.855;
	b = 58.585;
	p = r[7];
	q = 1 - r[7];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	CondomScaling = x;
	SexBehavParameters.out[CurrSim-1][7] = x;
}

void SimulateNGparameters()
{
	/*int ind, i, is;
	double x, y, a, b, p, q;
	double r[7]; // Random variables from U(0, 1)

	ind = 2;

	if(FixedUncertainty==0){
		int32 seed = time(0) + CurrSim;
		TRandomMersenne rg(seed);
		for(i=0; i<NGparameters.columns; i++){
			r[i] = rg.Random();
			RandomUniformNG.out[CurrSim-1][i] = r[i];
		}
	}
	else{
		for(i=0; i<NGparameters.columns; i++){
			r[i] = RandomUniformNG.out[CurrSim-1][i];}
	}

	// Simulate M->F transmission prob
	a = 1.433;
	b = 5.731;
	p = r[0];
	q = 1 - r[0];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	NGtransitionM.TransmProb = x;
	NGparameters.out[CurrSim-1][0] = x;

	NGtransitionM.RelTransmCSW = NGtransitionM.TransmProbSW/NGtransitionM.TransmProb;

	// Simulate F->M transmission prob
	a = 3.40;
	b = 24.93;
	p = r[1];
	q = 1 - r[1];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	NGtransitionF.TransmProb = x;
	NGparameters.out[CurrSim-1][1] = x;

	// Simulate % of male NG cases that become symptomatic
	a = 31.5;
	b = 3.5;
	p = r[2];
	q = 1 - r[2];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	NGtransitionM.SymptomaticPropn = x;
	NGparameters.out[CurrSim-1][2] = x;

	// Simulate % of female NG cases that become symptomatic
	a = 3.867;
	b = 5.8;
	p = r[3];
	q = 1 - r[3];
	cdfbet(&ind,&p,&q,&x,&y,&a,&b,0,0);
	NGtransitionF.SymptomaticPropn = x;
	NGparameters.out[CurrSim-1][3] = x;

	// Simulate the average duration of untreated NG in males
	a = 9.0;
	b = 0.6;
	p = r[4];
	q = 1 - r[4];
	cdfgam(&ind,&p,&q,&x,&a,&b,0,0);
	NGtransitionM.AveDuration[0] = x;
	NGtransitionM.AveDuration[1] = x;
	NGparameters.out[CurrSim-1][4] = x;

	// Simulate the average duration of untreated NG in females
	a = 9.0;
	b = 0.6;
	p = r[5];
	q = 1 - r[5];
	cdfgam(&ind,&p,&q,&x,&a,&b,0,0);
	NGtransitionF.AveDuration[0] = x;
	NGtransitionF.AveDuration[1] = x;
	NGparameters.out[CurrSim-1][5] = x;

	// Simulate the variance of the study effect
	a = 2.699;
	b = 11.735;
	p = r[6];
	q = 1 - r[6];
	cdfgam(&ind,&p,&q,&x,&a,&b,0,0);
	NGtransitionM.HouseholdLogL.VarStudyEffect = pow(x, 2.0);
	NGtransitionF.HouseholdLogL.VarStudyEffect = pow(x, 2.0);
	NGtransitionF.ANClogL.VarStudyEffect = pow(x, 2.0);
	NGtransitionF.FPClogL.VarStudyEffect = pow(x, 2.0);
	NGtransitionF.CSWlogL.VarStudyEffect = pow(x, 2.0);
	NGparameters.out[CurrSim-1][6] = pow(x, 2.0);*/
}

void GenerateSample()
{
	int i, j, iy;
	double CumTotL[simulations+1], r[samplesize];
	double lookup;
	double x1, DoubleIntDif1, intpart1;
	double MaxLogL;

	clock_t start2, finish2;
	double elapsed_time2;
	start2 = clock();

	ErrorCount = 0;
	CurrSim = 0;
	for(i=1; i<=simulations; i++){
		CurrSim += 1;
		x1 = CurrSim/1000.0;
		DoubleIntDif1 = modf(x1, &intpart1);
		if(DoubleIntDif1==0.0){
			finish2 = clock();
			elapsed_time2 = finish2 - start2;
			cout<<"Time to "<<i<<"th iteration: "<<elapsed_time2<<endl;
		}

		CurrYear = StartYear;
		if(i==1){
			ReadAllInputFiles();}
		else{
			ReadStartProfile("StartProfile.txt");
			BaselineCondomUse = BaselineCondomSvy;
		}
		do{
			ErrorInd = 0;
			SimulateParameters();
			GetAllPartnerRates();
			GetStartProfile();
		}
		while(ErrorInd==1);
		MaleHigh.Reset();
		MaleLow.Reset();
		FemHigh.Reset();
		FemLow.Reset();
		MaleChild.Reset();
		FemChild.Reset();
		GetAllNumbersBySTDstage();
		GetSummary();
		SetCalibParameters();
	
		for(iy=0; iy<ProjectionTerm; iy++){
			OneYearProj();
			if(KeepCurrYearFixed==0){
				CurrYear += 1;}
			if(HIVstageSumM[2][1]>=0){// Choice of HIVstageSumM[2][1] is arbitrary
				continue;}
			else{
				ErrorInd = 1;
				break;}
		}
		if(ErrorInd==0){
			CalcTotalLogL();
			if(SexCalib==1){
				CalcSumSquares();
				TotalLogL += SumSquares;
			}
			LogL.out[i-1][0] = TotalLogL;
			if(TotalLogL<0 || TotalLogL>=0){
				continue;}
			else{
				break;}
		}
		else{
			LogL.out[i-1][0] = -10000.0;} // -1000 is arbitrary; I chose it because it was lower
										 // than all the other log L values - it might be 
										 // necessary to lower it for certain runs.
	}
	cout<<"Initial set of parameter combinations evaluated."<<endl;

	// Calculate cumulative likelihood for each simulation
	CumTotL[0] = 0;
	MaxLogL = LogL.out[0][0];
	for(i=0; i<simulations; i++){
		if(LogL.out[i][0] > MaxLogL){
			MaxLogL = LogL.out[i][0];}
	}
	for(i=0; i<simulations; i++){
		CumTotL[i+1] = CumTotL[i] + exp(LogL.out[i][0] - MaxLogL);}
	
	// Generate random variables from the uniform (0, 1) distribution
	int32 seed = time(0);
	TRandomMersenne rg(seed);
	for(i=0; i<samplesize; i++){
		r[i] = rg.Random();}

	// Sample from the simulations and record ID numbers of the sampled simulations
	// in the sampleid array
	for(i=0; i<samplesize; i++){
		lookup = r[i]*CumTotL[simulations];
		for(j=0; j<simulations; j++){
			if(lookup>=CumTotL[j] && lookup<CumTotL[j+1]){
				sampleid[i] = j;
				break;
			}
		}
	}
	cout<<"Generated sample from initial set of parameter combinations."<<endl;

	LogL.Record("LogL.txt", 8);
	if(HIVcalib==1){
		HIVtransmProb.Record("HIVtransmProb.txt", 8);
		RandomUniformHIV.SampleInput();
		RandomUniformHIV.RecordSample("RandomUniformHIV.txt", 8);
	}
	if(SexCalib==1){
		SexBehavParameters.Record("SexBehavParameters.txt", 8);
		RandomUniformSex.SampleInput();
		RandomUniformSex.RecordSample("RandomUniformSex.txt", 8);
	}
	/*if(NGcalib==1){
		NGparameters.Record("NGparameters.txt", 8);
		RandomUniformNG.SampleInput();
		RandomUniformNG.RecordSample("RandomUniformNG.txt", 8);
	}*/
}

void RunSample()
{
	// Remember to set FixedUncertainty to 1 before calling this function.

	ifstream file1, file2;
	char filout[18];
	int i, c, iy, ia, ig, idum;

	// Read in random numbers for each parameter combination in sample
	if(HIVcalib==1){
		file1.open("RandomUniformHIV.txt");
		for(i=0; i<samplesize; i++){
			file1>>idum>>idum;
			for(c=0; c<RandomUniformHIV.columns; c++){
				file1>>RandomUniformHIV.out[i][c];}
		}
		file1.close();
	}
	if(SexCalib==1){
		file2.open("RandomUniformSex.txt");
		for(i=0; i<samplesize; i++){
			file2>>idum>>idum;
			for(c=0; c<RandomUniformSex.columns; c++){
				file2>>RandomUniformSex.out[i][c];}
		}
		file2.close();
	}

	// Run the model for each of the sampled parameter combinations
	CurrSim = 0;
	for(i=1; i<=samplesize; i++){
		CurrSim += 1;
		CurrYear = StartYear;
		if(i==1){
			ReadAllInputFiles();}
		else{
			ReadStartProfile("StartProfile.txt");
			BaselineCondomUse = BaselineCondomSvy;
		}
		SimulateParameters();

		GetAllPartnerRates();
		GetStartProfile();
		MaleHigh.Reset();
		MaleLow.Reset();
		FemHigh.Reset();
		FemLow.Reset();
		MaleChild.Reset();
		FemChild.Reset();
		GetAllNumbersBySTDstage();
		GetSummary();
		SetCalibParameters();
	
		for(iy=0; iy<ProjectionTerm; iy++){
			OneYearProj();
			if(KeepCurrYearFixed==0){
				CurrYear += 1;}
		}
		CalcTotalLogL(); 
		OutLogLStats.out[CurrSim-1][0] = HIVtransitionF.AntenatalNlogL.LogL;
		OutLogLStats.out[CurrSim-1][1] = HIVtransitionF.HouseholdNlogL.LogL + 
			HIVtransitionM.HouseholdNlogL.LogL;
		if(SexCalib==1){CalcSumSquares();}
		
		// Calculate remaining outputs
		/*for(iy=0; iy<9; iy++){
			OutANCprev15.out[CurrSim-1][iy] = HIVtransitionF.AntenatalNlogL.ModelPrev[iy*5];
			OutANCprev20.out[CurrSim-1][iy] = HIVtransitionF.AntenatalNlogL.ModelPrev[iy*5+1];
			OutANCprev25.out[CurrSim-1][iy] = HIVtransitionF.AntenatalNlogL.ModelPrev[iy*5+2];
			OutANCprev30.out[CurrSim-1][iy] = HIVtransitionF.AntenatalNlogL.ModelPrev[iy*5+3];
			OutANCprev35.out[CurrSim-1][iy] = HIVtransitionF.AntenatalNlogL.ModelPrev[iy*5+4];
		}*/
		for(ia=0; ia<9; ia++){
			OutHSRCprevM.out[CurrSim-1][ia] = HIVtransitionM.HouseholdNlogL.ModelPrev[ia];
			OutHSRCprevF.out[CurrSim-1][ia] = HIVtransitionF.HouseholdNlogL.ModelPrev[ia];
			OutHSRC2008M.out[CurrSim-1][ia] = HIVtransitionM.HouseholdNlogL.ModelPrev[ia+9];
			OutHSRC2008F.out[CurrSim-1][ia] = HIVtransitionF.HouseholdNlogL.ModelPrev[ia+9];
		}
		/*for(ig=0; ig<2; ig++){
			OutSexBias.out[CurrSim-1][ig] = ConcurrencyBias[0][ig];
			OutSexBias.out[CurrSim-1][ig+2] = ConcurrencyBias[1][ig];
			OutSexBias.out[CurrSim-1][ig+4] = AbstinenceBias[ig];
		}
		for(ig=0; ig<2; ig++){
			for(ia=0; ia<10; ia++){
				OutPopByAge.out[CurrSim-1][ig*10+ia] = TotalPopSum[ia][ig];
				OutVirgin.out[CurrSim-1][ig*10+ia] = VirginsSum[ia][ig];
				OutUnmarried0.out[CurrSim-1][ig*10+ia] = TotalPopSum[ia][ig] - 
					VirginsSum[ia][ig] - MarriedSum[ia][ig] - UnmarriedActiveSum[ia][ig];
				OutUnmarried1.out[CurrSim-1][ig*10+ia] = UnmarriedActiveSum[ia][ig] -
					UnmarriedMultSum[ia][ig];
				OutUnmarried2.out[CurrSim-1][ig*10+ia] = UnmarriedMultSum[ia][ig];
				OutMarried1.out[CurrSim-1][ig*10+ia] = MarriedSum[ia][ig] - 
					(MultPartnerSum[ia][ig] - UnmarriedMultSum[ia][ig]);
				OutMarried2.out[CurrSim-1][ig*10+ia] = MultPartnerSum[ia][ig] - 
					UnmarriedMultSum[ia][ig];
			}
		}*/
	}

	// Write to text files
	OutTotalPop.RecordSample("TotalPop.txt");
	OutTotalHIV.RecordSample("TotalHIV.txt");
	OutANCbias.RecordSample("ANCbias.txt");
	//OutANCbiasTrend.RecordSample("ANCbiasTrend.txt");
	OutModelVarANC.RecordSample("ModelVarANC.txt");
	OutLogLStats.RecordSample("LogLStats.txt");
	//OutModelVarHH.RecordSample("ModelVarHH.txt");
	//OutModelVarSex.RecordSample("ModelVarSex.txt");
	OutANCprev15.RecordSample("ANCprev15.txt");
	OutANCprev20.RecordSample("ANCprev20.txt");
	OutANCprev25.RecordSample("ANCprev25.txt");
	OutANCprev30.RecordSample("ANCprev30.txt");
	OutANCprev35.RecordSample("ANCprev35.txt");
	OutANCprevTot.RecordSample("ANCprevTot.txt");
	OutHSRCprevM.RecordSample("HSRCprevM.txt");
	OutHSRCprevF.RecordSample("HSRCprevF.txt");
	OutHSRC2002M.RecordSample("HSRC2002M.txt");
	OutHSRC2002F.RecordSample("HSRC2002F.txt");
	OutHSRC2008M.RecordSample("HSRC2008M.txt");
	OutHSRC2008F.RecordSample("HSRC2008F.txt");
	/*OutRHRUprev.RecordSample("RHRUprev.txt");
	OutSexBias.RecordSample("SexBias.txt");
	OutMarried1996.RecordSample("Married1996.txt");
	OutMarried2001.RecordSample("Married2001.txt");
	OutMarried2007.RecordSample("Married2007.txt");
	OutPartnerCalib.RecordSample("PartnerCalib.txt");
	Out15to49prevM.RecordSample("15to49prevM.txt");
	Out15to49prevF.RecordSample("15to49prevF.txt");
	OutCSWprev.RecordSample("CSWprev.txt");
	OutPopByAge.RecordSample("PopByAge.txt");
	OutVirgin.RecordSample("Virgin.txt");
	OutUnmarried0.RecordSample("Unmarried0.txt");
	OutUnmarried1.RecordSample("Unmarried1.txt");
	OutUnmarried2.RecordSample("Unmarried2.txt");
	OutMarried1.RecordSample("Married1.txt");
	OutMarried2.RecordSample("Married2.txt");*/
	OutNewHIV.RecordSample("NewHIV.txt");
	OutCoVHIVinc.RecordSample("CoVHIVinc.txt");
	/*OutHIVinc15to24F.RecordSample("HIVinc15F.txt");
	OutHIVinc25to49F.RecordSample("HIVinc25F.txt");
	OutHIVinc15to24M.RecordSample("HIVinc15M.txt");
	OutHIVinc25to49M.RecordSample("HIVinc25M.txt");*/
	OutHIVinc15to49.RecordSample("HIVinc15to49.txt");
	OutHIVincByAgeF.RecordSample("HIVincByAgeF.txt");
	OutHIVincByAgeM.RecordSample("HIVincByAgeM.txt");
	/*OutNewHIVMSM.RecordSample("NewHIVMSM.txt");
	OutNewHIVMSF.RecordSample("NewHIVMSF.txt");
	OutNewHIVMLM.RecordSample("NewHIVMLM.txt");
	OutNewHIVMLF.RecordSample("NewHIVMLF.txt");
	OutNewHIVUSM.RecordSample("NewHIVUSM.txt");
	OutNewHIVUSF.RecordSample("NewHIVUSF.txt");
	OutNewHIVCSM.RecordSample("NewHIVCSM.txt");
	OutNewHIVCSF.RecordSample("NewHIVCSF.txt");*/
	/*OutNewHIVMM.RecordSample("NewHIVMM.txt");
	OutNewHIVUM.RecordSample("NewHIVUM.txt");
	OutNewHIVMF.RecordSample("NewHIVMF.txt");
	OutNewHIVUF.RecordSample("NewHIVUF.txt");
	OutHIVnegMM.RecordSample("HIVnegMM.txt");
	OutHIVnegUM.RecordSample("HIVnegUM.txt");
	OutHIVnegMF.RecordSample("HIVnegMF.txt");
	OutHIVnegUF.RecordSample("HIVnegUF.txt");*/
	//OutAIDSdeaths.RecordSample("AIDSdeaths.txt");
}

void ReadStartingPoints()
{
	int ir;
	ifstream file;

	file.open("StartingPoints.txt");
	if(file==0)cout<<"File open error"<<endl;
	for(ir=0; ir<NumberSeries; ir++){
		file>>lMtoFtransmST.out[0][ir]>>lMtoFtransmLT.out[0][ir]>>
			lFtoMtransmST.out[0][ir]>>lFtoMtransmLT.out[0][ir]>>lInitHIVprevHigh.out[0][ir]>>
			lRelPartnerAcqM.out[0][ir]>>lRelPartnerAcqF.out[0][ir]>>
			lAssortativeness.out[0][ir]>>lRelPartnerAcqMM.out[0][ir]>>
			lRelPartnerAcqMF.out[0][ir]>>lRelPartnerAcqLM.out[0][ir]>>
			lRelPartnerAcqLF.out[0][ir]>>lCondomBias.out[0][ir]>>
			lRelARTinfectiousness.out[0][ir];
	}
	file.close();
}

void ReadCovariance(const char* input)
{
	int ir, ic;
	ifstream file;

	file.open(input);
	if(file==0)cout<<"File open error"<<endl;
	for(ir=0; ir<MCMCdim; ir++){
		for(ic=0; ic<MCMCdim; ic++){
			file>>Covariance[ir][ic];}
	}
	file.close();
}

void ReadOnePrior(ifstream* file, MCMCparameter* a)
{
	*file>>a->index>>a->PriorType>>a->Param1>>a->Param2;
}

void ReadPriors()
{
	ifstream file;

	file.open("Priors.txt");
	if(file==0)cout<<"File open error"<<endl;
	//ReadOnePrior(&file, &lMtoFtransmCSW);
	ReadOnePrior(&file, &lMtoFtransmST);
	ReadOnePrior(&file, &lMtoFtransmLT);
	//ReadOnePrior(&file, &lFtoMtransmCSW);
	ReadOnePrior(&file, &lFtoMtransmST);
	ReadOnePrior(&file, &lFtoMtransmLT);
	ReadOnePrior(&file, &lInitHIVprevHigh);
	//ReadOnePrior(&file, &lHighPropnM);
	//ReadOnePrior(&file, &lHighPropnF);
	ReadOnePrior(&file, &lRelPartnerAcqM);
	ReadOnePrior(&file, &lRelPartnerAcqF);
	ReadOnePrior(&file, &lAssortativeness);
	//ReadOnePrior(&file, &lMeanPartnerDur);
	//ReadOnePrior(&file, &lMeanPartnerGapM);
	//ReadOnePrior(&file, &lMeanPartnerGapF);
	ReadOnePrior(&file, &lRelPartnerAcqMM);
	ReadOnePrior(&file, &lRelPartnerAcqMF);
	ReadOnePrior(&file, &lRelPartnerAcqLM);
	ReadOnePrior(&file, &lRelPartnerAcqLF);
	ReadOnePrior(&file, &lCondomBias);
	ReadOnePrior(&file, &lRelARTinfectiousness);
	file.close();
}

void GetCholesky()
{
	// This method for calculating the Cholesky decomposition is from Healy M.J.R. (1986) 
	// "Matrices for Statistics" Clarendon Press, Oxford. (See pp. 54-5)

	int ir, ic, ii;
	double sumprod;
	int PositiveDefinite = 1; // 0 if Covariance matrix is NOT positive definite (default
							  // assumption is that it is positive definite)

	for(ir=0; ir<MCMCdim; ir++){
		if(PositiveDefinite==1){
			for(ic=0; ic<MCMCdim; ic++){
				if(ir>ic){
					Cholesky[ir][ic] = 0.0;}
				if(ir==ic){
					sumprod = 0.0;
					if(ir>0){
						for(ii=0; ii<ir; ii++){
							sumprod += pow(Cholesky[ii][ic], 2.0);}
					}
					if(Covariance[ir][ic]>sumprod){
						Cholesky[ir][ic] = sqrt(Covariance[ir][ic] - sumprod);}
					else{
						PositiveDefinite = 0;
						break;
					}
				}
				if(ir<ic){
					sumprod = 0.0;
					if(ir>0){
						for(ii=0; ii<ir; ii++){
							sumprod += Cholesky[ii][ir] * Cholesky[ii][ic];}
					}
					Cholesky[ir][ic] = (Covariance[ir][ic] - sumprod)/Cholesky[ir][ir];
				}
			}
		}
	}
	if(PositiveDefinite==0){
		// Cholesky decomposition fails, so don't induce any correlation
		for(ir=0; ir<MCMCdim; ir++){
			for(ic=0; ic<MCMCdim; ic++){
				if(ir==ic){
					Cholesky[ir][ic] = sqrt(Covariance[ir][ic]);}
				else{
					Cholesky[ir][ic] = 0.0;}
			}
		}
	}
}

void NextMCstep(int ir, int ic)
{
	int ind, i, j, is, iy;
	int counter; // CumIterations - 1
	double x, y, a, b, p, q;
	double r[MCMCdim+1]; // Random variables from U(0, 1), which get converted to std normal
	double Proposal[MCMCdim]; // Proposed parameter combination
	double lPropPrior; // Log of the prior distribution evaluated at the proposal
	double lPropPosterior; // Log of the posterior distribution evaluated at the proposal
	double AcceptanceProb; // Prob of accepting the proposed parameter combination

	ind = 2;
	if(CumIterations>0){
		counter = CumIterations - 1;}
	else{
		counter = 0;}

	// Randomly generate a parameter combination (Proposal)
	if(CumIterations>0){
		int32 seed = time(0) + CumIterations * NumberSeries + ic + ErrorCount;
		TRandomMersenne rg(seed);
		for(i=0; i<MCMCdim+1; i++){
			r[i] = rg.Random();}
		for(i=0; i<MCMCdim; i++){
			a = 0.0; // mean of std normal
			b = 1.0; // std deviation of std normal
			p = r[i];
			q = 1 - r[i];
			cdfnor(&ind,&p,&q,&x,&a,&b,0,0);
			r[i] = x;
		}
		cout.precision(10);
		for(i=0; i<MCMCdim; i++){
			Proposal[i] = 0.0;
			for(j=0; j<MCMCdim; j++){
				Proposal[i] += r[j] * Cholesky[j][i];}
		}
	}
	else{
		for(i=0; i<MCMCdim; i++){
			Proposal[i] = 0.0;}
	}
	//Proposal[0] += lMtoFtransmCSW.out[counter][ic];
	Proposal[0] += lMtoFtransmST.out[counter][ic];
	Proposal[1] += lMtoFtransmLT.out[counter][ic];
	//Proposal[3] += lFtoMtransmCSW.out[counter][ic];
	Proposal[2] += lFtoMtransmST.out[counter][ic];
	Proposal[3] += lFtoMtransmLT.out[counter][ic];
	Proposal[4] += lInitHIVprevHigh.out[counter][ic];
	//Proposal[7] += lHighPropnM.out[counter][ic];
	//Proposal[8] += lHighPropnF.out[counter][ic];
	Proposal[5] += lRelPartnerAcqM.out[counter][ic];
	Proposal[6] += lRelPartnerAcqF.out[counter][ic];
	Proposal[7] += lAssortativeness.out[counter][ic];
	//Proposal[13] += lMeanPartnerDur.out[counter][ic];
	//Proposal[14] += lMeanPartnerGapM.out[counter][ic];
	//Proposal[15] += lMeanPartnerGapF.out[counter][ic];
	Proposal[8] += lRelPartnerAcqMM.out[counter][ic];
	Proposal[9] += lRelPartnerAcqMF.out[counter][ic];
	Proposal[10] += lRelPartnerAcqLM.out[counter][ic];
	Proposal[11] += lRelPartnerAcqLF.out[counter][ic];
	Proposal[12] += lCondomBias.out[counter][ic];
	Proposal[13] += lRelARTinfectiousness.out[counter][ic];

	// Calculate log of the prior distribution
	//lPropPrior = lMtoFtransmCSW.GetLogPrior(Proposal[0]);
	lPropPrior = lMtoFtransmST.GetLogPrior(Proposal[0]);
	lPropPrior += lMtoFtransmLT.GetLogPrior(Proposal[1]);
	//lPropPrior += lFtoMtransmCSW.GetLogPrior(Proposal[3]);
	lPropPrior += lFtoMtransmST.GetLogPrior(Proposal[2]);
	lPropPrior += lFtoMtransmLT.GetLogPrior(Proposal[3]);
	lPropPrior += lInitHIVprevHigh.GetLogPrior(Proposal[4]);
	//lPropPrior += lHighPropnM.GetLogPrior(Proposal[7]);
	//lPropPrior += lHighPropnF.GetLogPrior(Proposal[8]);
	lPropPrior += lRelPartnerAcqM.GetLogPrior(Proposal[5]);
	lPropPrior += lRelPartnerAcqF.GetLogPrior(Proposal[6]);
	lPropPrior += lAssortativeness.GetLogPrior(Proposal[7]);
	//lPropPrior += lMeanPartnerDur.GetLogPrior(Proposal[13]);
	//lPropPrior += lMeanPartnerGapM.GetLogPrior(Proposal[14]);
	//lPropPrior += lMeanPartnerGapF.GetLogPrior(Proposal[15]);
	lPropPrior += lRelPartnerAcqMM.GetLogPrior(Proposal[8]);
	lPropPrior += lRelPartnerAcqMF.GetLogPrior(Proposal[9]);
	lPropPrior += lRelPartnerAcqLM.GetLogPrior(Proposal[10]);
	lPropPrior += lRelPartnerAcqLF.GetLogPrior(Proposal[11]);
	lPropPrior += lCondomBias.GetLogPrior(Proposal[12]);
	lPropPrior += lRelARTinfectiousness.GetLogPrior(Proposal[13]);

	// Calculate log of the likelihood (code copied & adapted from GenerateSample function)
	CurrYear = StartYear;
	if(ir==0){
		ReadAllInputFiles();}
	else{
		BaselineCondomUse = BaselineCondomSvy;
		ReadStartProfile("StartProfile.txt");
	}
	ErrorInd = 0;
	//HIVtransitionM.TransmProb[0] = 1.0/(1.0 + exp(-Proposal[0]));
	InitHIVtransm[1][0] = 1.0/(1.0 + exp(-Proposal[0]));
	InitHIVtransm[2][0] = 1.0/(1.0 + exp(-Proposal[1]));
	//HIVtransitionF.TransmProb[0] = 1.0/(1.0 + exp(-Proposal[3]));
	InitHIVtransm[1][1] = 1.0/(1.0 + exp(-Proposal[2]));
	InitHIVtransm[2][1] = 1.0/(1.0 + exp(-Proposal[3]));
	HIVtransitionM.TransmProb[0] = InitHIVtransm[0][0];
	HIVtransitionF.TransmProb[0] = InitHIVtransm[0][1];
	for(is=0; is<3; is++){
		HIVtransitionM.TransmProb[is+1] = InitHIVtransm[1][0];
		HIVtransitionM.TransmProb[is+4] = InitHIVtransm[2][0];
		HIVtransitionF.TransmProb[is+1] = InitHIVtransm[1][1];
		HIVtransitionF.TransmProb[is+4] = InitHIVtransm[2][1];
	}
	if(CofactorType==0){
		HIVtransitionM.TransmProb[1] *= 2.0;
		HIVtransitionM.TransmProb[3] *= 0.5;
		HIVtransitionM.TransmProb[4] *= 2.0;
		HIVtransitionM.TransmProb[6] *= 0.5;
		HIVtransitionF.TransmProb[1] *= 2.0;
		HIVtransitionF.TransmProb[3] *= 0.5;
		HIVtransitionF.TransmProb[4] *= 2.0;
		HIVtransitionF.TransmProb[6] *= 0.5;
	}
	HIVtransitionM.HIVinfecIncrease[4] = 5.0/(1.0 + exp(-Proposal[13])) - 1.0;
	HIVtransitionF.HIVinfecIncrease[4] = 5.0/(1.0 + exp(-Proposal[13])) - 1.0;
	InitHIVprevHigh = 1.0/(1.0 + exp(-Proposal[4]));
	//HighPropnM = 1.0/(1.0 + exp(-Proposal[7]));
	//HighPropnF = 1.0/(1.0 + exp(-Proposal[8]));
	PartnerEffectNew[0][0] = 1.0/(1.0 + exp(-Proposal[5]));
	PartnerEffectNew[0][1] = 1.0/(1.0 + exp(-Proposal[6]));
	AssortativeM = 1.0/(1.0 + exp(-Proposal[7]));
	AssortativeF  = AssortativeM;
	//BaselineCondomUse *= exp(Proposal[12]);
	//MeanDurSTrel[0][0] = exp(Proposal[13]);
	//MeanDurSTrel[0][1] = MeanDurSTrel[0][0];
	//MeanDurSTrel[1][0] = MeanDurSTrel[0][0];
	//MeanDurSTrel[1][1] = MeanDurSTrel[0][0];
	//BasePartnerAcqH[0] = exp(-Proposal[14]); // Note that we take the inverse of the mean gap
	//BasePartnerAcqH[1] = exp(-Proposal[15]);
	PartnerEffectNew[1][0] = 1.0/(1.0 + exp(-Proposal[8]));
	PartnerEffectNew[1][1] = 1.0/(1.0 + exp(-Proposal[9]));
	PartnershipFormation[1][0] = 1.0/(1.0 + exp(-Proposal[10]));
	PartnershipFormation[1][1] = 1.0/(1.0 + exp(-Proposal[11]));
	CondomScaling = 1.0/(1.0 + exp(-Proposal[12]));
	GetAllPartnerRates();
	GetStartProfile();
	if(ErrorInd==0){
		MaleHigh.Reset();
		MaleLow.Reset();
		FemHigh.Reset();
		FemLow.Reset();
		MaleChild.Reset();
		FemChild.Reset();
		GetAllNumbersBySTDstage();
		GetSummary();
		SetCalibParameters();
	
		for(iy=0; iy<ProjectionTerm; iy++){
			OneYearProj();
			if(KeepCurrYearFixed==0){
				CurrYear += 1;}
			if(ErrorInd==0){
				continue;}
			else{
				break;}
		}
		if(ErrorInd==0){
			CalcTotalLogL();
			if(SexCalib==1){
				CalcSumSquares();
				TotalLogL += SumSquares;
			}
		}
		else{
			TotalLogL = -1000.0;} // Arbitrary value, low enough to prevent sampling
		if(TotalLogL<=0 || TotalLogL>0){
			iy = 0;}
		else{ // Code to tell us if there is an error (& associated parameter combination)
			for(i=0; i<MCMCdim; i++){
				cout<<"Proposal["<<i<<"]: "<<Proposal[i]<<endl;}
		}
	}
	else{
		TotalLogL = -1000.0;} // Arbitrary value, low enough to prevent sampling
	
	// Calculate log of the posterior distribution and decide whether to accept proposal
	lPropPosterior = lPropPrior + TotalLogL;
	if(CumIterations>0){
		AcceptanceProb = exp(lPropPosterior - lPosterior.out[counter][ic]);}
	else{
		AcceptanceProb = 1.0;}
	if(AcceptanceProb>=r[MCMCdim]){
		AcceptInd.out[CumIterations][ic] = 1.0;
		lPosterior.out[CumIterations][ic] = lPropPosterior;
		//lMtoFtransmCSW.out[CumIterations][ic] = Proposal[0];
		lMtoFtransmST.out[CumIterations][ic] = Proposal[0];
		lMtoFtransmLT.out[CumIterations][ic] = Proposal[1];
		//lFtoMtransmCSW.out[CumIterations][ic] = Proposal[3];
		lFtoMtransmST.out[CumIterations][ic] = Proposal[2];
		lFtoMtransmLT.out[CumIterations][ic] = Proposal[3];
		lInitHIVprevHigh.out[CumIterations][ic] = Proposal[4];
		//lHighPropnM.out[CumIterations][ic] = Proposal[7];
		//lHighPropnF.out[CumIterations][ic] = Proposal[8];
		lRelPartnerAcqM.out[CumIterations][ic] = Proposal[5];
		lRelPartnerAcqF.out[CumIterations][ic] = Proposal[6];
		lAssortativeness.out[CumIterations][ic] = Proposal[7];
		//lMeanPartnerDur.out[CumIterations][ic] = Proposal[13];
		//lMeanPartnerGapM.out[CumIterations][ic] = Proposal[14];
		//lMeanPartnerGapF.out[CumIterations][ic] = Proposal[15];
		lRelPartnerAcqMM.out[CumIterations][ic] = Proposal[8];
		lRelPartnerAcqMF.out[CumIterations][ic] = Proposal[9];
		lRelPartnerAcqLM.out[CumIterations][ic] = Proposal[10];
		lRelPartnerAcqLF.out[CumIterations][ic] = Proposal[11];
		lCondomBias.out[CumIterations][ic] = Proposal[12];
		lRelARTinfectiousness.out[CumIterations][ic] = Proposal[13];
	}
	else{
		AcceptInd.out[CumIterations][ic] = 0.0;
		lPosterior.out[CumIterations][ic] = lPosterior.out[counter][ic];
		//lMtoFtransmCSW.out[CumIterations][ic] = lMtoFtransmCSW.out[counter][ic];
		lMtoFtransmST.out[CumIterations][ic] = lMtoFtransmST.out[counter][ic];
		lMtoFtransmLT.out[CumIterations][ic] = lMtoFtransmLT.out[counter][ic];
		//lFtoMtransmCSW.out[CumIterations][ic] = lFtoMtransmCSW.out[counter][ic];
		lFtoMtransmST.out[CumIterations][ic] = lFtoMtransmST.out[counter][ic];
		lFtoMtransmLT.out[CumIterations][ic] = lFtoMtransmLT.out[counter][ic];
		lInitHIVprevHigh.out[CumIterations][ic] = lInitHIVprevHigh.out[counter][ic];
		//lHighPropnM.out[CumIterations][ic] = lHighPropnM.out[counter][ic];
		//lHighPropnF.out[CumIterations][ic] = lHighPropnF.out[counter][ic];
		lRelPartnerAcqM.out[CumIterations][ic] = lRelPartnerAcqM.out[counter][ic];
		lRelPartnerAcqF.out[CumIterations][ic] = lRelPartnerAcqF.out[counter][ic];
		lAssortativeness.out[CumIterations][ic] = lAssortativeness.out[counter][ic];
		//lMeanPartnerDur.out[CumIterations][ic] = lMeanPartnerDur.out[counter][ic];
		//lMeanPartnerGapM.out[CumIterations][ic] = lMeanPartnerGapM.out[counter][ic];
		//lMeanPartnerGapF.out[CumIterations][ic] = lMeanPartnerGapF.out[counter][ic];
		lRelPartnerAcqMM.out[CumIterations][ic] = lRelPartnerAcqMM.out[counter][ic];
		lRelPartnerAcqMF.out[CumIterations][ic] = lRelPartnerAcqMF.out[counter][ic];
		lRelPartnerAcqLM.out[CumIterations][ic] = lRelPartnerAcqLM.out[counter][ic];
		lRelPartnerAcqLF.out[CumIterations][ic] = lRelPartnerAcqLF.out[counter][ic];
		lCondomBias.out[CumIterations][ic] = lCondomBias.out[counter][ic];
		lRelARTinfectiousness.out[CumIterations][ic] = lRelARTinfectiousness.out[counter][ic];
	}
}

void GetAcceptanceRate(int n)
{
	int i, j;

	AcceptanceRate = 0.0;
	for(i=CumIterations-n; i<CumIterations; i++){
		for(j=0; j<NumberSeries; j++){
			AcceptanceRate += AcceptInd.out[i][j];}
	}
	AcceptanceRate = AcceptanceRate/(n * NumberSeries);
}

void TuneJumpingDbn(int n)
{
	int i, j;

	GetAcceptanceRate(n);

	if(AcceptanceRate<0.1){
		JumpingDbnVarAdj = 0.25;}
	else if(AcceptanceRate>0.4){
		JumpingDbnVarAdj = 4.0;}
	else{
		JumpingDbnVarAdj = 1.0;}

	for(i=0; i<MCMCdim; i++){
		for(j=0; j<MCMCdim; j++){
			Covariance[i][j] *= JumpingDbnVarAdj;}
	}
	GetCholesky();
}

void UpdateCovar(int n)
{
	int i, j;

	lPosterior.GetMean(n);
	//lMtoFtransmCSW.GetMean(n);
	lMtoFtransmST.GetMean(n);
	lMtoFtransmLT.GetMean(n);
	//lFtoMtransmCSW.GetMean(n);
	lFtoMtransmST.GetMean(n);
	lFtoMtransmLT.GetMean(n);
	lInitHIVprevHigh.GetMean(n);
	//lHighPropnM.GetMean(n);
	//lHighPropnF.GetMean(n);
	lRelPartnerAcqM.GetMean(n);
	lRelPartnerAcqF.GetMean(n);
	lAssortativeness.GetMean(n);
	//lMeanPartnerDur.GetMean(n);
	//lMeanPartnerGapM.GetMean(n);
	//lMeanPartnerGapF.GetMean(n);
	lRelPartnerAcqMM.GetMean(n);
	lRelPartnerAcqMF.GetMean(n);
	lRelPartnerAcqLM.GetMean(n);
	lRelPartnerAcqLF.GetMean(n);
	lCondomBias.GetMean(n);
	lRelARTinfectiousness.GetMean(n);

	lPosterior.GetAllCovar(n);
	//lMtoFtransmCSW.GetAllCovar(n);
	lMtoFtransmST.GetAllCovar(n);
	lMtoFtransmLT.GetAllCovar(n);
	//lFtoMtransmCSW.GetAllCovar(n);
	lFtoMtransmST.GetAllCovar(n);
	lFtoMtransmLT.GetAllCovar(n);
	lInitHIVprevHigh.GetAllCovar(n);
	//lHighPropnM.GetAllCovar(n);
	//lHighPropnF.GetAllCovar(n);
	lRelPartnerAcqM.GetAllCovar(n);
	lRelPartnerAcqF.GetAllCovar(n);
	lAssortativeness.GetAllCovar(n);
	//lMeanPartnerDur.GetAllCovar(n);
	//lMeanPartnerGapM.GetAllCovar(n);
	//lMeanPartnerGapF.GetAllCovar(n);
	lRelPartnerAcqMM.GetAllCovar(n);
	lRelPartnerAcqMF.GetAllCovar(n);
	lRelPartnerAcqLM.GetAllCovar(n);
	lRelPartnerAcqLF.GetAllCovar(n);
	lCondomBias.GetAllCovar(n);
	lRelARTinfectiousness.GetAllCovar(n);

	JumpingDbnVarAdj = pow(2.4, 2.0)/MCMCdim;
	for(i=0; i<MCMCdim; i++){
		for(j=0; j<MCMCdim; j++){
			Covariance[i][j] *= JumpingDbnVarAdj;}
	}
	GetCholesky();
}

void CalcConvergence(double pi)
{
	lPosterior.TestConvergence(pi);
	//lMtoFtransmCSW.TestConvergence(pi);
	lMtoFtransmST.TestConvergence(pi);
	lMtoFtransmLT.TestConvergence(pi);
	//lFtoMtransmCSW.TestConvergence(pi);
	lFtoMtransmST.TestConvergence(pi);
	lFtoMtransmLT.TestConvergence(pi);
	lInitHIVprevHigh.TestConvergence(pi);
	//lHighPropnM.TestConvergence(pi);
	//lHighPropnF.TestConvergence(pi);
	lRelPartnerAcqM.TestConvergence(pi);
	lRelPartnerAcqF.TestConvergence(pi);
	lAssortativeness.TestConvergence(pi);
	//lMeanPartnerDur.TestConvergence(pi);
	//lMeanPartnerGapM.TestConvergence(pi);
	//lMeanPartnerGapF.TestConvergence(pi);
	lRelPartnerAcqMM.TestConvergence(pi);
	lRelPartnerAcqMF.TestConvergence(pi);
	lRelPartnerAcqLM.TestConvergence(pi);
	lRelPartnerAcqLF.TestConvergence(pi);
	lCondomBias.TestConvergence(pi);
	lRelARTinfectiousness.TestConvergence(pi);
}

void RecordCovariance()
{
	int i, j;
	ofstream file("CovarianceFinal.txt");

	for(i=0; i<MCMCdim; i++){
		for(j=0; j<MCMCdim; j++){
			file<<"	"<<setw(10)<<right<<Covariance[i][j];}
		file<<endl;
	}
	file.close();
}

void Metropolis()
{
	// Before running this function, make sure that the RunLength is less than the specified
	// value of the simulations variable (this determines the dimension of the output array).

	int i, j, LengthCovarEvaln;
	double x1, x2, DoubleIntDif1, DoubleIntDif2, intpart1, intpart2;

	clock_t start2, finish2;
	double elapsed_time2;
	start2 = clock();

	// Get initial Cholesky decomposition
	JumpingDbnVarAdj = pow(2.4, 2.0)/MCMCdim;
	ReadStartingPoints();
	ReadCovariance("Covariance.txt");
	ReadPriors();
	for(i=0; i<MCMCdim; i++){
		for(j=0; j<MCMCdim; j++){
			Covariance[i][j] *= JumpingDbnVarAdj;}
	}
	GetCholesky();

	// Do n iterations (n = RunLength)
	ErrorCount = 0;
	CumIterations = 0;
	for(i=0; i<RunLength; i++){
		for(j=0; j<NumberSeries; j++){
			NextMCstep(i, j);}
		CumIterations += 1;
		x1 = CumIterations/TuningFreq;
		DoubleIntDif1 = modf(x1, &intpart1);
		x2 = CumIterations/UpdateCovarFreq;
		DoubleIntDif2 = modf(x2, &intpart2);
		if(DoubleIntDif2==0.0 && CumIterations<=UpdateCovarPeriod){
			LengthCovarEvaln = CumIterations * 0.5; // 0.5 is arbitrary
			UpdateCovar(LengthCovarEvaln);
		}
		else{
			if(DoubleIntDif1==0.0 && CumIterations<=TuningPeriod){
				TuneJumpingDbn(TuningFreq);}
		}
		x1 = CumIterations/100.0;
		DoubleIntDif1 = modf(x1, &intpart1);
		if(DoubleIntDif1==0.0){
			finish2 = clock();
			elapsed_time2 = finish2 - start2;
			cout<<"Time to "<<CumIterations<<"th iteration: "<<elapsed_time2<<endl;
		}
	}

	// Record the results and the Covariance matrix
	//lMtoFtransmCSW.Record("lMtoFtransmCSW.txt", 8);
	lMtoFtransmST.Record("lMtoFtransmST.txt", 8);
	lMtoFtransmLT.Record("lMtoFtransmLT.txt", 8);
	//lFtoMtransmCSW.Record("lFtoMtransmCSW.txt", 8);
	lFtoMtransmST.Record("lFtoMtransmST.txt", 8);
	lFtoMtransmLT.Record("lFtoMtransmLT.txt", 8);
	lInitHIVprevHigh.Record("lInitHIVprevHigh.txt", 8);
	//lHighPropnM.Record("lHighPropnM.txt", 8);
	//lHighPropnF.Record("lHighPropnF.txt", 8);
	lRelPartnerAcqM.Record("lRelPartnerAcqM.txt", 8);
	lRelPartnerAcqF.Record("lRelPartnerAcqF.txt", 8);
	lAssortativeness.Record("lAssortativeness.txt", 8);
	//lMeanPartnerDur.Record("lMeanPartnerDur.txt", 8);
	//lMeanPartnerGapM.Record("lMeanPartnerGapM.txt", 8);
	//lMeanPartnerGapF.Record("lMeanPartnerGapF.txt", 8);
	lRelPartnerAcqMM.Record("lRelPartnerAcqMM.txt", 8);
	lRelPartnerAcqMF.Record("lRelPartnerAcqMF.txt", 8);
	lRelPartnerAcqLM.Record("lRelPartnerAcqLM.txt", 8);
	lRelPartnerAcqLF.Record("lRelPartnerAcqLF.txt", 8);
	lCondomBias.Record("lCondomBias.txt", 8);
	lRelARTinfectiousness.Record("lRelARTinfectiousness.txt", 8);
	lPosterior.Record("lPosterior.txt", 8);
	AcceptInd.Record("AcceptInd.txt", 8);
	RecordCovariance();

	// Check acceptance rate and convergence criteria
	GetAcceptanceRate(50);
	cout<<"AcceptanceRate: "<<AcceptanceRate<<endl;
	CalcConvergence(0.5);
}

void MetropolisCont(int n)
{
	// Before running this function, make sure that the projected cumulative iterations by
	// the end of the run does not exceed the specified value of the simulations variable.

	int i, j, LengthCovarEvaln;
	double x1, x2, DoubleIntDif1, DoubleIntDif2, intpart1, intpart2;
	
	clock_t start2, finish2;
	double elapsed_time2;
	start2 = clock();

	CumIterations = n;

	// Read in parameter combinations generated so far
	//lMtoFtransmCSW.ReadCumIterations("lMtoFtransmCSW.txt");
	lMtoFtransmST.ReadCumIterations("lMtoFtransmST.txt");
	lMtoFtransmLT.ReadCumIterations("lMtoFtransmLT.txt");
	//lFtoMtransmCSW.ReadCumIterations("lFtoMtransmCSW.txt");
	lFtoMtransmST.ReadCumIterations("lFtoMtransmST.txt");
	lFtoMtransmLT.ReadCumIterations("lFtoMtransmLT.txt");
	lInitHIVprevHigh.ReadCumIterations("lInitHIVprevHigh.txt");
	//lHighPropnM.ReadCumIterations("lHighPropnM.txt");
	//lHighPropnF.ReadCumIterations("lHighPropnF.txt");
	lRelPartnerAcqM.ReadCumIterations("lRelPartnerAcqM.txt");
	lRelPartnerAcqF.ReadCumIterations("lRelPartnerAcqF.txt");
	lAssortativeness.ReadCumIterations("lAssortativeness.txt");
	//lMeanPartnerDur.ReadCumIterations("lMeanPartnerDur.txt");
	//lMeanPartnerGapM.ReadCumIterations("lMeanPartnerGapM.txt");
	//lMeanPartnerGapF.ReadCumIterations("lMeanPartnerGapF.txt");
	lRelPartnerAcqMM.ReadCumIterations("lRelPartnerAcqMM.txt");
	lRelPartnerAcqMF.ReadCumIterations("lRelPartnerAcqMF.txt");
	lRelPartnerAcqLM.ReadCumIterations("lRelPartnerAcqLM.txt");
	lRelPartnerAcqLF.ReadCumIterations("lRelPartnerAcqLF.txt");
	lCondomBias.ReadCumIterations("lCondomBias.txt");
	lRelARTinfectiousness.ReadCumIterations("lRelARTinfectiousness.txt");
	lPosterior.ReadCumIterations("lPosterior.txt");
	AcceptInd.ReadCumIterations("AcceptInd.txt");

	// Get Cholesky decomposition (note that we don't have to apply the JumpingDbnVarAdj
	// here, since Covariance matrix already has it 'built in')
	ReadCovariance("CovarianceFinal.txt");
	ReadPriors();
	GetCholesky();

	// Rest of the code is almost identical to that in Metropolis function (main exception:
	// we don't set CumIterations to 0)

	// Do n iterations (n = RunLength)
	ErrorCount = 0;
	for(i=0; i<RunLength; i++){
		for(j=0; j<NumberSeries; j++){
			NextMCstep(i, j);}
		CumIterations += 1;
		x1 = CumIterations/TuningFreq;
		DoubleIntDif1 = modf(x1, &intpart1);
		x2 = CumIterations/UpdateCovarFreq;
		DoubleIntDif2 = modf(x2, &intpart2);
		if(DoubleIntDif2==0.0 && CumIterations<=UpdateCovarPeriod){
			LengthCovarEvaln = CumIterations * 0.5; // 0.5 is arbitrary
			UpdateCovar(LengthCovarEvaln);
		}
		else{
			if(DoubleIntDif1==0.0 && CumIterations<=TuningPeriod){
				TuneJumpingDbn(TuningFreq);}
		}
		x1 = CumIterations/100.0;
		DoubleIntDif1 = modf(x1, &intpart1);
		if(DoubleIntDif1==0.0){
			finish2 = clock();
			elapsed_time2 = finish2 - start2;
			cout<<"Time to "<<CumIterations<<"th iteration: "<<elapsed_time2<<endl;
		}
	}

	// Record the results and the Covariance matrix
	//lMtoFtransmCSW.Record("lMtoFtransmCSW.txt", 8);
	lMtoFtransmST.Record("lMtoFtransmST.txt", 8);
	lMtoFtransmLT.Record("lMtoFtransmLT.txt", 8);
	//lFtoMtransmCSW.Record("lFtoMtransmCSW.txt", 8);
	lFtoMtransmST.Record("lFtoMtransmST.txt", 8);
	lFtoMtransmLT.Record("lFtoMtransmLT.txt", 8);
	lInitHIVprevHigh.Record("lInitHIVprevHigh.txt", 8);
	//lHighPropnM.Record("lHighPropnM.txt", 8);
	//lHighPropnF.Record("lHighPropnF.txt", 8);
	lRelPartnerAcqM.Record("lRelPartnerAcqM.txt", 8);
	lRelPartnerAcqF.Record("lRelPartnerAcqF.txt", 8);
	lAssortativeness.Record("lAssortativeness.txt", 8);
	//lMeanPartnerDur.Record("lMeanPartnerDur.txt", 8);
	//lMeanPartnerGapM.Record("lMeanPartnerGapM.txt", 8);
	//lMeanPartnerGapF.Record("lMeanPartnerGapF.txt", 8);
	lRelPartnerAcqMM.Record("lRelPartnerAcqMM.txt", 8);
	lRelPartnerAcqMF.Record("lRelPartnerAcqMF.txt", 8);
	lRelPartnerAcqLM.Record("lRelPartnerAcqLM.txt", 8);
	lRelPartnerAcqLF.Record("lRelPartnerAcqLF.txt", 8);
	lCondomBias.Record("lCondomBias.txt", 8);
	lRelARTinfectiousness.Record("lRelARTinfectiousness.txt", 8);
	lPosterior.Record("lPosterior.txt", 8);
	AcceptInd.Record("AcceptInd.txt", 8);
	RecordCovariance();

	// Check acceptance rate and convergence criteria
	GetAcceptanceRate(50);
	cout<<"AcceptanceRate: "<<AcceptanceRate<<endl;
	CalcConvergence(0.5);
}

void SetCalibParameters()
{
	double AveOR;

	AveOR = 0.68; // Posterior mean from the ASSA2002 uncertainty analysis

	// Values in the 5 lines below are from the simulateparameters function in the C++
	// version of the ASSA2002 model (see derivation in section 3.3.1.3 of the uncertainty
	// analysis report)
	HIVtransitionF.AntenatalNlogL.BiasMult[1] = 0.885450/(AveOR*0.044021 + 0.841431);
	HIVtransitionF.AntenatalNlogL.BiasMult[2] = 0.902648/(AveOR*0.080730 + 0.810868);
	HIVtransitionF.AntenatalNlogL.BiasMult[3] = 0.870821/(AveOR*0.108702 + 0.707487);
	HIVtransitionF.AntenatalNlogL.BiasMult[4] = 0.921259/(AveOR*0.142296 + 0.698347);
	HIVtransitionF.AntenatalNlogL.BiasMult[5] = 0.937727/(AveOR*0.117837 + 0.784181);

	// FPC weights are the proportions of women currently using any modern contraceptive
	// method, as estimated from the 1998 DHS. By assuming the same proportions for single
	// women and women currently in partnerships, we may be under-estimating STD prevalence,
	// but by setting the proportions to 0 in the <15 and 50+ age categories, we may be
	// over-estimating STD prevalence. So the biases cancel out to some extent.
	FPCweights[0] = 0.0;
	FPCweights[1] = 0.285;
	FPCweights[2] = 0.565;
	FPCweights[3] = 0.578;
	FPCweights[4] = 0.586;
	FPCweights[5] = 0.558;
	FPCweights[6] = 0.502;
	FPCweights[7] = 0.380;
	FPCweights[8] = 0.0;

	// Setting the default values for the variance of the study effects (only relevant to
	// the sentinel surveillance data)
	if(HIVcalib==1 && HIVtransitionF.ANClogL.VarStudyEffect==0){
		HIVtransitionM.SetVarStudyEffect(0.0529);
		HIVtransitionF.SetVarStudyEffect(0.0529);
	}
	if(HSVcalib==1 && HSVtransitionF.ANClogL.VarStudyEffect==0){
		HSVtransitionM.SetVarStudyEffect(0.0529);
		HSVtransitionF.SetVarStudyEffect(0.0529);
	}
	if(TPcalib==1 && TPtransitionF.ANClogL.VarStudyEffect==0){
		TPtransitionM.SetVarStudyEffect(0.0529);
		TPtransitionF.SetVarStudyEffect(0.0529);
	}
	if(HDcalib==1 && HDtransitionF.ANClogL.VarStudyEffect==0){
		HDtransitionM.SetVarStudyEffect(0.0529);
		HDtransitionF.SetVarStudyEffect(0.0529);
	}
	if(NGcalib==1 && NGtransitionF.ANClogL.VarStudyEffect==0){
		NGtransitionM.SetVarStudyEffect(0.0529);
		NGtransitionF.SetVarStudyEffect(0.0529);
	}
	if(CTcalib==1 && CTtransitionF.ANClogL.VarStudyEffect==0){
		CTtransitionM.SetVarStudyEffect(0.0529);
		CTtransitionF.SetVarStudyEffect(0.0529);
	}
	if(TVcalib==1 && TVtransitionF.ANClogL.VarStudyEffect==0){
		TVtransitionM.SetVarStudyEffect(0.0529);
		TVtransitionF.SetVarStudyEffect(0.0529);
	}
	if(BVcalib==1 && BVtransitionF.ANClogL.VarStudyEffect==0){
		BVtransitionF.SetVarStudyEffect(0.0529);}
	if(VCcalib==1 && VCtransitionF.ANClogL.VarStudyEffect==0){
		VCtransitionF.SetVarStudyEffect(0.0529);}
}

void CalcTotalLogL()
{
	TotalLogL = 0.0;
	if(HIVcalib==1){
		HIVtransitionF.CSWlogL.CalcLogL();
		HIVtransitionF.AntenatalNlogL.CalcLogL();
		HIVtransitionF.HouseholdNlogL.CalcLogL();
		HIVtransitionM.HouseholdNlogL.CalcLogL();
		TotalLogL += HIVtransitionF.CSWlogL.LogL + HIVtransitionF.AntenatalNlogL.LogL + 
			HIVtransitionF.HouseholdNlogL.LogL + HIVtransitionM.HouseholdNlogL.LogL;
	}
	if(HSVcalib==1){
		HSVtransitionF.ANClogL.CalcLogL();
		HSVtransitionF.FPClogL.CalcLogL();
		HSVtransitionF.CSWlogL.CalcLogL();
		HSVtransitionF.GUDlogL.CalcLogL();
		HSVtransitionF.HouseholdLogL.CalcLogL();
		HSVtransitionM.GUDlogL.CalcLogL();
		HSVtransitionM.HouseholdLogL.CalcLogL();
		TotalLogL += HSVtransitionF.ANClogL.LogL + HSVtransitionF.FPClogL.LogL +
			HSVtransitionF.CSWlogL.LogL + HSVtransitionF.GUDlogL.LogL +
			HSVtransitionF.HouseholdLogL.LogL + HSVtransitionM.GUDlogL.LogL +
			HSVtransitionM.HouseholdLogL.LogL;
	}
	if(TPcalib==1){
		// Note that I'm assuming we would not include the GUD data.
		TPtransitionF.AntenatalNlogL.CalcLogL();
		TPtransitionF.ANClogL.CalcLogL();
		TPtransitionF.FPClogL.CalcLogL();
		TPtransitionF.CSWlogL.CalcLogL();
		TPtransitionF.HouseholdLogL.CalcLogL();
		TPtransitionM.HouseholdLogL.CalcLogL();
		TotalLogL += TPtransitionF.AntenatalNlogL.LogL + TPtransitionF.ANClogL.LogL + 
			TPtransitionF.FPClogL.LogL + TPtransitionF.CSWlogL.LogL +
			TPtransitionF.HouseholdLogL.LogL + TPtransitionM.HouseholdLogL.LogL;
	}
	if(HDcalib==1){
		HDtransitionF.GUDlogL.CalcLogL();
		HDtransitionM.GUDlogL.CalcLogL();
		TotalLogL += HDtransitionF.GUDlogL.LogL + HDtransitionM.GUDlogL.LogL;
	}
	if(NGcalib==1){
		NGtransitionF.ANClogL.CalcLogL();
		NGtransitionF.FPClogL.CalcLogL();
		NGtransitionF.CSWlogL.CalcLogL();
		NGtransitionF.HouseholdLogL.CalcLogL();
		NGtransitionM.HouseholdLogL.CalcLogL();
		TotalLogL += NGtransitionF.ANClogL.LogL + NGtransitionF.FPClogL.LogL +
			NGtransitionF.CSWlogL.LogL + NGtransitionF.HouseholdLogL.LogL +
			NGtransitionM.HouseholdLogL.LogL;
	}
	if(CTcalib==1){
		CTtransitionF.ANClogL.CalcLogL();
		CTtransitionF.FPClogL.CalcLogL();
		CTtransitionF.CSWlogL.CalcLogL();
		CTtransitionF.HouseholdLogL.CalcLogL();
		CTtransitionM.HouseholdLogL.CalcLogL();
		TotalLogL += CTtransitionF.ANClogL.LogL + CTtransitionF.FPClogL.LogL +
			CTtransitionF.CSWlogL.LogL + CTtransitionF.HouseholdLogL.LogL +
			CTtransitionM.HouseholdLogL.LogL;
	}
	if(TVcalib==1){
		TVtransitionF.ANClogL.CalcLogL();
		TVtransitionF.FPClogL.CalcLogL();
		TVtransitionF.CSWlogL.CalcLogL();
		TVtransitionF.HouseholdLogL.CalcLogL();
		TVtransitionM.HouseholdLogL.CalcLogL();
		TotalLogL += TVtransitionF.ANClogL.LogL + TVtransitionF.FPClogL.LogL +
			TVtransitionF.CSWlogL.LogL + TVtransitionF.HouseholdLogL.LogL +
			TVtransitionM.HouseholdLogL.LogL;
	}
	if(BVcalib==1){
		BVtransitionF.ANClogL.CalcLogL();
		BVtransitionF.FPClogL.CalcLogL();
		BVtransitionF.CSWlogL.CalcLogL();
		TotalLogL += BVtransitionF.ANClogL.LogL + BVtransitionF.FPClogL.LogL +
			BVtransitionF.CSWlogL.LogL;
	}
	if(VCcalib==1){
		VCtransitionF.ANClogL.CalcLogL();
		VCtransitionF.FPClogL.CalcLogL();
		VCtransitionF.CSWlogL.CalcLogL();
		TotalLogL += VCtransitionF.ANClogL.LogL + VCtransitionF.FPClogL.LogL +
			VCtransitionF.CSWlogL.LogL;
	}
}

double ReturnNegLogL(double ParameterSet[10])
{
	int ip;
	double HIVandSexLogL;

	CurrYear = StartYear;
	ReadAllInputFiles();
	InitHIVprevHigh = ParameterSet[0];
	for(ip=0; ip<3; ip++){
		HIVtransitionM.TransmProb[ip] = ParameterSet[ip+1];}
	for(ip=0; ip<3; ip++){
		HIVtransitionF.TransmProb[ip] = ParameterSet[ip+4];}

	if(SexCalib==1){GetAllPartnerRates();}
	GetStartProfile();
	MaleHigh.Reset();
	MaleLow.Reset();
	FemHigh.Reset();
	FemLow.Reset();
	MaleChild.Reset();
	FemChild.Reset();
	GetAllNumbersBySTDstage();
	GetSummary();
	SetCalibParameters();
	
	int iy;
	for(iy=0; iy<ProjectionTerm; iy++){
		OneYearProj();
		if(KeepCurrYearFixed==0){
			CurrYear += 1;}
	}
	CalcTotalLogL();
	HIVandSexLogL = TotalLogL;
	if(SexHIVcount > 1){
		CalcSumSquares();
		HIVandSexLogL += SumSquares; // Shock! Horror! Actually the 'SumSquares' variable
									 // is a log likelihood statistic, so don't be alarmed.
	}

	return -HIVandSexLogL;
}

void ReadInitSimplex(const char* input, double ParameterCombinations[11][10], int Dimension)
{
	int ir, ic;
	double sumvertices;
	ifstream file;

	file.open(input);
	if(file==0)cout<<"File open error"<<endl;
	for(ir=0; ir<Dimension+1; ir++){
		for(ic=0; ic<Dimension; ic++){
			file>>ParameterCombinations[ir][ic];}
	}
	file.close();

	// To avoid a situation in which the algorithm converges on a zero parameter value
	// that then becomes 'fixed' at zero, we restart with parameter values slightly above
	// zero. If zero is indeed the true parameter value, the algorithm should return you
	// to zero fairly quickly.
	for(ic=0; ic<Dimension; ic++){
		sumvertices = 0;
		for(ir=0; ir<Dimension+1; ir++){
			sumvertices += ParameterCombinations[ir][ic];}
		if(sumvertices==0){
			for(ir=0; ir<Dimension+1; ir++){
				ParameterCombinations[ir][ic] = 0.0001/(ir + 1.0);}
		}
	}

	if(SexHIVcount>0){
		for(ir=0; ir<Dimension+1; ir++){
			for(ic=0; ic<Dimension; ic++){
				ParameterCombinations[ir][ic] *= RandomAdjHIV[ir][ic];}
			if(ParameterCombinations[ir][3] > ParameterCombinations[ir][2]){
				ParameterCombinations[ir][3] = ParameterCombinations[ir][2];}
			if(ParameterCombinations[ir][6] > ParameterCombinations[ir][5]){
				ParameterCombinations[ir][6] = ParameterCombinations[ir][5];}
		}
	}
}

void SaveFinalSimplex(const char* filout, double ParameterCombinations[11][10], int Dimension)
{
	int ic, ir;
	ofstream file(filout);

	for(ir=0; ir<Dimension+1; ir++){
		for(ic=0; ic<Dimension; ic++){
			file<<"	"<<setw(15)<<right<<ParameterCombinations[ir][ic];}
		file<<endl;
	}
	file.close();
}

void MaximizeLikelihood(double FTol, const char* input, const char* filout)
{
	// This function implements the Downhill Simplex Method, and is copied from the 
	// AMOEBA function outlined on pp. 292-3 of Press et al, 1986, Numerical Recipes,
	// Cambridge, Cambridge University Press. I have changed some of the variable and
	// function names to make it easier to understand. For example, the ReturnNegLogL()
	// function takes the place of the FUNK() function (see further comments below).
	// Note that this function minimizes the NEGATIVE of the log likelihood, which is
	// equivalent to maximizing the log likelihood.
	// Note that one cannot maximize the likelihood if there are more than 10 parameters;
	// if there are more than 10, you need to change the dimensions of the arrays below.
	// Also note that I've modified the algorithm to prevent any of the parameters lying
	// outside the interval [0, 1]. This restriction might not be appropriate for certain
	// parameters, but it is only at the reflection and extrapolation steps that the 
	// function has been modified, so it is easy to change back to the original. I've also
	// modified the code at the reflection and extrapolation steps to prevent the
	// transmission prob in marital relationships exceeding that in non-marital relationships.

	// VERY IMPORTANT: Always check the FinalSimplex output file before concluding that
	// you have reached convergence. The log likelihood values might all be very close
	// together, but unless the parameters are also all close to one another, it does not
	// mean that convergence has been reached. Reduce FTol and run the algorithm again,
	// replacing "InitialSimplex.txt" with "FinalSimplex.txt", and replacing
	// "FinalSimplex.txt" with "FinalSimplex2.txt". (Incidentally, you can use the same
	// method if you reach the maximum number of iterations (500) and wish to continue.)

	int ic, ir, it; // Counters for parameters, vertices and iterations respectively
	int Dimension = 7; // Takes the place of NDIM in AMOEBA
	int Vertices = Dimension + 1; // Takes the place of MPTS in AMOEBA
	int offset;
	double NegLogL[11]; // Takes the place of Y[] in AMOEBA
	double Alpha = 1.0;
	double Beta = 0.5;
	double Gamma = 2.0;
	double ParameterCombinations[11][10]; // Takes the place of matrix P in AMOEBA
	double AveParam[10]; // Takes the place of PBAR in AMOEBA
	double AltParam1[10]; // Takes the place of PR in AMOEBA
	double AltParam2[10]; // Takes the place of PRR in AMOEBA
	double AltLogL1; // Takes the place of YPR in AMOEBA
	double AltLogL2; // Takes the place of YPRR in AMOEBA
	int iLowest, iHighest, i2ndHighest; // Take the place of ILO, IHI, INHI
	double RTol;
	int MaxIterations;

	// Determine the starting values of ParameterCombinations and NegLogL arrays
	ReadInitSimplex(input, ParameterCombinations, Dimension);
	for(ir=0; ir<Vertices; ir++){
		for(ic=0; ic<Dimension; ic++){
			AltParam1[ic] = ParameterCombinations[ir][ic];}
		NegLogL[ir] = ReturnNegLogL(AltParam1);
	}

	// Determine MaxIterations
	if(SexHIVcount<1){
		MaxIterations = 500;}
	else{
		MaxIterations = 10000;}

	//Determine the starting values of iLowest, iHighest, i2ndHighest, RTol
	iLowest = 0;
	if(NegLogL[0] > NegLogL[1]){
		iHighest = 0;
		i2ndHighest = 1;
	}
	else{
		iHighest = 1;
		i2ndHighest = 0;
	}
	for(ir=0; ir<Vertices; ir++){
		if(NegLogL[ir] < NegLogL[iLowest]){
			iLowest = ir;}
		if(NegLogL[ir] > NegLogL[iHighest]){
			i2ndHighest = iHighest;
			iHighest = ir;
		}
		else if(NegLogL[ir] > NegLogL[i2ndHighest]){
			if(ir!=iHighest){
				i2ndHighest = ir;}
		}
	}
	RTol = 2.0 * fabs(NegLogL[iHighest] - NegLogL[iLowest])/(fabs(NegLogL[iHighest]) +
		fabs(NegLogL[iLowest]));

	// Iterate until convergence is achieved
	it = 0;
	while(RTol > FTol && it < MaxIterations){
		it += 1;
		for(ic=0; ic<Dimension; ic++){ // Calculate average
			AveParam[ic] = 0;}
		for(ir=0; ir<Vertices; ir++){
			if(ir!=iHighest){
				for(ic=0; ic<Dimension; ic++){
					AveParam[ic] += ParameterCombinations[ir][ic];}
			}
		}
		for(ic=0; ic<Dimension; ic++){
			AveParam[ic] = AveParam[ic]/Dimension;
			AltParam1[ic] = (1.0 + Alpha) * AveParam[ic] - Alpha *
				ParameterCombinations[iHighest][ic]; // Reflection step
			if(AltParam1[ic]<0.0){AltParam1[ic]=0.0;}
			if(AltParam1[ic]>1.0){AltParam1[ic]=1.0;}
		}
		if(AltParam1[3]>AltParam1[2]){AltParam1[3]=AltParam1[2];}
		if(AltParam1[6]>AltParam1[5]){AltParam1[6]=AltParam1[5];}
		AltLogL1 = ReturnNegLogL(AltParam1);
		if(AltLogL1 <= NegLogL[iLowest]){
			for(ic=0; ic<Dimension; ic++){ // Extrapolation step
				AltParam2[ic] = Gamma * AltParam1[ic] + (1.0 - Gamma) * AveParam[ic];
				if(AltParam2[ic]<0.0){AltParam2[ic]=0.0;}
				if(AltParam2[ic]>1.0){AltParam2[ic]=1.0;}
			}
			if(AltParam2[3]>AltParam2[2]){AltParam2[3]=AltParam2[2];}
			if(AltParam2[6]>AltParam2[5]){AltParam2[6]=AltParam2[5];}
			AltLogL2 = ReturnNegLogL(AltParam2);
			if(AltLogL2 < NegLogL[iLowest]){
				for(ic=0; ic<Dimension; ic++){
					ParameterCombinations[iHighest][ic] = AltParam2[ic];}
				NegLogL[iHighest] = AltLogL2;
			}
			else{
				for(ic=0; ic<Dimension; ic++){
					ParameterCombinations[iHighest][ic] = AltParam1[ic];}
				NegLogL[iHighest] = AltLogL1;
			}
		}
		else if(AltLogL1 >= NegLogL[i2ndHighest]){
			if(AltLogL1 < NegLogL[iHighest]){
				for(ic=0; ic<Dimension; ic++){
					ParameterCombinations[iHighest][ic] = AltParam1[ic];}
				NegLogL[iHighest] = AltLogL1;
			}
			for(ic=0; ic<Dimension; ic++){ // Contraction step
				AltParam2[ic] = Beta * ParameterCombinations[iHighest][ic] + (1.0 - Beta) *
					AveParam[ic];
				// Not necessary to check if value is in [0, 1] range
			}
			AltLogL2 = ReturnNegLogL(AltParam2);
			if(AltLogL2 < NegLogL[iHighest]){
				for(ic=0; ic<Dimension; ic++){
					ParameterCombinations[iHighest][ic] = AltParam2[ic];}
				NegLogL[iHighest] = AltLogL2;
			}
			else{
				for(ir=0; ir<Vertices; ir++){
					if(ir!=iLowest){
						for(ic=0; ic<Dimension; ic++){
							AltParam1[ic] = 0.5 * (ParameterCombinations[ir][ic] +
								ParameterCombinations[iLowest][ic]);
							// Not necessary to check if value is in [0, 1] range
							ParameterCombinations[ir][ic] = AltParam1[ic];
						}
						NegLogL[ir] = ReturnNegLogL(AltParam1);
					}
				}
			}
		}
		else{
			for(ic=0; ic<Dimension; ic++){
				ParameterCombinations[iHighest][ic] = AltParam1[ic];}
			NegLogL[iHighest] = AltLogL1;
		}

		// Lastly, update iLowest, iHighest, i2ndHighest, RTol (same code as before)
		iLowest = 0;
		if(NegLogL[0] > NegLogL[1]){
			iHighest = 0;
			i2ndHighest = 1;
		}
		else{
			iHighest = 1;
			i2ndHighest = 0;
		}
		for(ir=0; ir<Vertices; ir++){
			if(NegLogL[ir] < NegLogL[iLowest]){
				iLowest = ir;}
			if(NegLogL[ir] > NegLogL[iHighest]){
				i2ndHighest = iHighest;
				iHighest = ir;
			}
			else if(NegLogL[ir] > NegLogL[i2ndHighest]){
				if(ir!=iHighest){
					i2ndHighest = ir;}
			}
		}
		RTol = 2.0 * fabs(NegLogL[iHighest] - NegLogL[iLowest])/(fabs(NegLogL[iHighest]) +
			fabs(NegLogL[iLowest]));
	}

	// Generate outputs
	SaveFinalSimplex(filout, ParameterCombinations, Dimension);

	if(SexHIVcount==0){
		cout.precision(10);
		for(ir=0; ir<Vertices; ir++){
			cout<<"NegLogL["<<ir<<"]: "<<NegLogL[ir]<<endl;}
		for(ic=0; ic<Dimension; ic++){
			cout<<"ML estimate of parameter "<<ic+1<<": "<<ParameterCombinations[iLowest][ic]<<endl;}
		cout<<"Number of iterations: "<<it<<endl;
	}
	if(SexHIVcount>0){
		for(ic=0; ic<Dimension; ic++){
			MLEforHIV[ic] = ParameterCombinations[iLowest][ic];}
		if(SexHIVcount==1){
			for(ir=0; ir<Vertices; ir++){
				FinalHIVlogL[0][ir] = NegLogL[ir];}
		}
		else{
			offset = SexHIVcount;
			SexHIVcount = 1;
			for(ir=0; ir<Vertices; ir++){
				FinalLogL[offset-1][ir] = NegLogL[ir];
				for(ic=0; ic<Dimension; ic++){
					AltParam1[ic] = ParameterCombinations[ir][ic];}
				FinalHIVlogL[offset-1][ir] = ReturnNegLogL(AltParam1);
			}
		}
	}
}

void SaveHIVcalibOutput(const char* filout)
{
	int i;
	ofstream file(filout);

	// First save the antenatal prevalence estimates
	file<<right<<"Antenatal prevalence estimates"<<endl;
	for(i=0; i<HIVtransitionF.AntenatalNlogL.Observations; i++){
		file<<right<<HIVtransitionF.AntenatalNlogL.ModelPrev[i]<<endl;}

	// Then save the female household prevalence survey estimates
	file<<right<<"Female household prevalence estimates"<<endl;
	for(i=0; i<HIVtransitionF.HouseholdNlogL.Observations; i++){
		file<<right<<HIVtransitionF.HouseholdNlogL.ModelPrev[i]<<endl;}

	// Thirdly, save the male household prevalence survey estimates
	file<<right<<"Male household prevalence estimates"<<endl;
	for(i=0; i<HIVtransitionM.HouseholdNlogL.Observations; i++){
		file<<right<<HIVtransitionM.HouseholdNlogL.ModelPrev[i]<<endl;}

	// Lastly, save the CSW prevalence estimates
	file<<right<<"CSW prevalence estimates"<<endl;
	for(i=0; i<HIVtransitionF.CSWlogL.Observations; i++){
		file<<right<<HIVtransitionF.CSWlogL.ModelPrev[i]<<endl;}
	file.close();
}

void SaveTrend(double Trend[41], const char* filout)
{
	int iy;
	ofstream file(filout);

	for(iy=0; iy<=CurrYear-StartYear; iy++){
		file<<right<<Trend[iy]<<endl;}
	file.close();
}

void RecordAllSTDpropns()
{
	ofstream file("STDpropns.txt");

	FemHigh.Virgin.RecordPropnsByStage(&file);
	FemLow.Virgin.RecordPropnsByStage(&file);
	MaleHigh.NoPartner.RecordPropnsByStage(&file);
	MaleHigh.S1.RecordPropnsByStage(&file);
	MaleHigh.S2.RecordPropnsByStage(&file);
	MaleHigh.L1.RecordPropnsByStage(&file);
	MaleHigh.L2.RecordPropnsByStage(&file);
	MaleHigh.S11.RecordPropnsByStage(&file);
	MaleHigh.S12.RecordPropnsByStage(&file);
	MaleHigh.S22.RecordPropnsByStage(&file);
	MaleHigh.L11.RecordPropnsByStage(&file);
	MaleHigh.L12.RecordPropnsByStage(&file);
	MaleHigh.L21.RecordPropnsByStage(&file);
	MaleHigh.L22.RecordPropnsByStage(&file);
	MaleLow.NoPartner.RecordPropnsByStage(&file);
	MaleLow.S1.RecordPropnsByStage(&file);
	MaleLow.S2.RecordPropnsByStage(&file);
	MaleLow.L1.RecordPropnsByStage(&file);
	MaleLow.L2.RecordPropnsByStage(&file);
	FemHigh.FSW.RecordPropnsByStage(&file);
	FemHigh.NoPartner.RecordPropnsByStage(&file);
	FemHigh.S1.RecordPropnsByStage(&file);
	FemHigh.S2.RecordPropnsByStage(&file);
	FemHigh.L1.RecordPropnsByStage(&file);
	FemHigh.L2.RecordPropnsByStage(&file);MaleHigh.S11.RecordPropnsByStage(&file);
	FemHigh.S12.RecordPropnsByStage(&file);
	FemHigh.S22.RecordPropnsByStage(&file);
	FemHigh.L11.RecordPropnsByStage(&file);
	FemHigh.L12.RecordPropnsByStage(&file);
	FemHigh.L21.RecordPropnsByStage(&file);
	FemHigh.L22.RecordPropnsByStage(&file);
	FemLow.NoPartner.RecordPropnsByStage(&file);
	FemLow.S1.RecordPropnsByStage(&file);
	FemLow.S2.RecordPropnsByStage(&file);
	FemLow.L1.RecordPropnsByStage(&file);
	FemLow.L2.RecordPropnsByStage(&file);
}

void CalcSumSquares()
{
	// Note that although we refer to the "sum of squares" statistic here and elsewhere, 
	// we're actually referring to a log likelihood statistic (as is evident from the code
	// below). In the earlier version (TSHISA 5), we used sum of square statistics, and to
	// maintain consistency I've kept the names of variables the same as before.

	int ia, ig;
	double VarLogitVirgin[2][2], VarLogitUMult[5][2], VarLogitMMult[5][2], 
		VarLogitUSingle[5][2]; // Variance of logit-transformed propns reporting a particular
							   // behaviour (sampling variation only)
	double ModelVarEst, SampleVarSum;

	SumSquares = 0.0;
	ModelVarEst = 0.0;
	SampleVarSum = 0.0;

	/*for(ia=0; ia<2; ia++){
		for(ig=0; ig<2; ig++){
			if(GetSDfromData==0){
				SumSquares += -0.5 * log(2.0 * 3.141592654) - log(VirginPropnSD[ia][ig]) - 0.5 * 
					pow((VirginPropnC[ia][ig] - VirginPropn[ia][ig])/VirginPropnSD[ia][ig], 2.0);}
			else{
				VarLogitVirgin[ia][ig] = pow(VirginPropnSD[ia][ig]/(VirginPropnC[ia][ig] * 
					(1.0 - VirginPropnC[ia][ig])), 2.0);
				ModelVarEst += pow(log(VirginPropnC[ia][ig]/(1.0 - VirginPropnC[ia][ig])) - 
					log(VirginPropn[ia][ig]/(1.0 - VirginPropn[ia][ig])), 2.0) - 
					VarLogitVirgin[ia][ig];
			}
		}
	}*/
	for(ia=0; ia<5; ia++){
		for(ig=0; ig<2; ig++){
			if(GetSDfromData==0){
				SumSquares += -0.5 * log(2.0 * 3.141592654) - log(UnmarriedMultPartnersSD[ia][ig])
					- 0.5 * pow((UnmarriedMultPartnersC[ia][ig] - UnmarriedMultPartners[ia][ig])/
					UnmarriedMultPartnersSD[ia][ig], 2.0);
				SumSquares += -0.5 * log(2.0 * 3.141592654) - log(MarriedMultPartnersSD[ia][ig])
					- 0.5 * pow((MarriedMultPartnersC[ia][ig] - MarriedMultPartners[ia][ig])/
					MarriedMultPartnersSD[ia][ig], 2.0);
				SumSquares += -0.5 * log(2.0 * 3.141592654) - log(UnmarriedSingleSD[ia][ig])
					- 0.5 * pow((UnmarriedSingleC[ia][ig] - UnmarriedSingle[ia][ig])/
					UnmarriedSingleSD[ia][ig], 2.0);
			}
			else{
				VarLogitUMult[ia][ig] = pow(UnmarriedMultPartnersSD[ia][ig]/
					(UnmarriedMultPartnersC[ia][ig] * (1.0 - UnmarriedMultPartnersC[ia][ig])), 
					2.0);
				ModelVarEst += pow(log(UnmarriedMultPartnersC[ia][ig]/(1.0 - 
					UnmarriedMultPartnersC[ia][ig])) - log(UnmarriedMultPartners[ia][ig]/
					(1.0 - UnmarriedMultPartners[ia][ig])), 2.0);
				VarLogitMMult[ia][ig] = pow(MarriedMultPartnersSD[ia][ig]/
					(MarriedMultPartnersC[ia][ig] * (1.0 - MarriedMultPartnersC[ia][ig])), 2.0);
				ModelVarEst += pow(log(MarriedMultPartnersC[ia][ig]/(1.0 - 
					MarriedMultPartnersC[ia][ig])) - log(MarriedMultPartners[ia][ig]/(1.0 - 
					MarriedMultPartners[ia][ig])), 2.0);
				VarLogitUSingle[ia][ig] = pow(UnmarriedSingleSD[ia][ig]/
					(UnmarriedSingleC[ia][ig] * (1.0 - UnmarriedSingleC[ia][ig])), 2.0);
				ModelVarEst += pow(log(UnmarriedSingleC[ia][ig]/(1.0 - 
					UnmarriedSingleC[ia][ig])) - log(UnmarriedSingle[ia][ig]/(1.0 - 
					UnmarriedSingle[ia][ig])), 2.0);
				SampleVarSum += VarLogitUMult[ia][ig] + VarLogitMMult[ia][ig] +
					VarLogitUSingle[ia][ig];
			}
		}
	}

	if(GetSDfromData==1){
		ModelVarEst = 0.0;
		/*ModelVarEst = ModelVarEst/(2.0 * 3.0 * (5.0 - 1.0)) - SampleVarSum/(2.0 * 3.0 * 5.0);
		if(ModelVarEst < 0.0){
			ModelVarEst = 0.0;}*/
		if(FixedUncertainty==1){OutModelVarSex.out[CurrSim-1][0] = ModelVarEst;}
		/*for(ia=0; ia<2; ia++){
			for(ig=0; ig<2; ig++){
				SumSquares += -0.5 * (log(2.0 * 3.141592654 * (VarLogitVirgin[ia][ig] + 
					ModelVarEst)) + pow(log(VirginPropnC[ia][ig]/(1.0 - VirginPropnC[ia][ig])) 
					- log(VirginPropn[ia][ig]/(1.0 - VirginPropn[ia][ig])), 2.0)/
					(VarLogitVirgin[ia][ig] + ModelVarEst));
			}
		}*/
		for(ia=0; ia<5; ia++){
			for(ig=0; ig<2; ig++){
				if(ig==0 || ia<4){ // Omit the zero counts of concurrent women aged 60+
					SumSquares += -0.5 * (log(2.0 * 3.141592654 * (VarLogitUMult[ia][ig] + 
						ModelVarEst)) + pow(log(UnmarriedMultPartnersC[ia][ig]/(1.0 - 
						UnmarriedMultPartnersC[ia][ig])) - log(UnmarriedMultPartners[ia][ig]/
						(1.0 - UnmarriedMultPartners[ia][ig])), 2.0)/(VarLogitUMult[ia][ig] + 
						ModelVarEst));
					SumSquares += -0.5 * (log(2.0 * 3.141592654 * (VarLogitMMult[ia][ig] + 
						ModelVarEst)) + pow(log(MarriedMultPartnersC[ia][ig]/(1.0 - 
						MarriedMultPartnersC[ia][ig])) - log(MarriedMultPartners[ia][ig]/
						(1.0 - MarriedMultPartners[ia][ig])), 2.0)/(VarLogitMMult[ia][ig] + 
						ModelVarEst));
				}
				SumSquares += -0.5 * (log(2.0 * 3.141592654 * (VarLogitUSingle[ia][ig] + 
					ModelVarEst)) + pow(log(UnmarriedSingleC[ia][ig]/(1.0 - 
					UnmarriedSingleC[ia][ig])) - log(UnmarriedSingle[ia][ig]/(1.0 - 
					UnmarriedSingle[ia][ig])), 2.0)/(VarLogitUSingle[ia][ig] + ModelVarEst));
			}
		}
	}

	// I haven't added code for the marriage data - since it's not appropriate to include 
	// this in the calibration (there are no marriage parameters being altered).
}

double ReturnSumSquares(double ParameterSet[16])
{
	int ia;

	CurrYear = StartYear;
	ReadAllInputFiles();
	
	// Overwrite the default sexual behaviour parameters
	HighPropnM = ParameterSet[0];
	HighPropnF = ParameterSet[1];
	PartnershipFormation[0][0] = ParameterSet[8];
	PartnershipFormation[1][0] = 1.0;
	PartnershipFormation[0][1] = ParameterSet[9];
	PartnershipFormation[1][1] = 1.0;
	for(ia=0; ia<16; ia++){
		AgeEffectPartners[ia][0] = ParameterSet[2] * pow(ParameterSet[4], ParameterSet[6]) *
			pow(5.0 * ia + 2.5, ParameterSet[6] - 1.0) * exp(-ParameterSet[4] * 
			(5.0 * ia + 2.5));
		AgeEffectPartners[ia][1] = ParameterSet[3] * pow(ParameterSet[5], ParameterSet[7]) *
			pow(5.0 * ia + 2.5, ParameterSet[7] - 1.0) * exp(-ParameterSet[5] * 
			(5.0 * ia + 2.5));
	}
	PartnerEffectNew[1][0] = ParameterSet[10];
	PartnerEffectNew[1][1] = ParameterSet[11];

	// Continue with the rest of the ResetAll function
	GetAllPartnerRates();
	GetStartProfile();
	MaleHigh.Reset();
	MaleLow.Reset();
	FemHigh.Reset();
	FemLow.Reset();
	MaleChild.Reset();
	FemChild.Reset();
	GetAllNumbersBySTDstage();
	GetSummary();

	SetCalibParameters();
	
	int iy;
	for(iy=0; iy<ProjectionTerm; iy++){
		OneYearProj();
		if(KeepCurrYearFixed==0){
			CurrYear += 1;}
	}
	CalcSumSquares();

	return -SumSquares;
}

void ReadInitSex(const char* input, double ParameterCombinations[17][16], int Dimension)
{
	int ir, ic;
	ifstream file;

	file.open(input);
	if(file==0)cout<<"File open error"<<endl;
	for(ir=0; ir<Dimension+1; ir++){
		for(ic=0; ic<Dimension; ic++){
			file>>ParameterCombinations[ir][ic];}
	}
	file.close();

	if(SexHIVcount>0){
		for(ir=0; ir<Dimension+1; ir++){
			for(ic=0; ic<Dimension; ic++){
				ParameterCombinations[ir][ic] *= RandomAdjSex[ir][ic];}
			if(ParameterCombinations[ir][0]>1.0){ParameterCombinations[ir][0] = 1.0;}
			if(ParameterCombinations[ir][1]>1.0){ParameterCombinations[ir][1] = 1.0;}
			if(ParameterCombinations[ir][8]<1.0){ParameterCombinations[ir][8] = 1.0;}
			if(ParameterCombinations[ir][9]<1.0){ParameterCombinations[ir][9] = 1.0;}
		}
	}
}

void SaveFinalSex(const char* filout, double ParameterCombinations[17][16], int Dimension)
{
	int ic, ir;
	ofstream file(filout);

	for(ir=0; ir<Dimension+1; ir++){
		for(ic=0; ic<Dimension; ic++){
			file<<"	"<<setw(15)<<right<<ParameterCombinations[ir][ic];}
		file<<endl;
	}
	file.close();
}

void MinimizeSumSquares(double FTol, const char* input, const char* filout)
{
	// This function implements the Downhill Simplex Method, and is copied from the 
	// AMOEBA function outlined on pp. 292-3 of Press et al, 1986, Numerical Recipes,
	// Cambridge, Cambridge University Press. I have changed some of the variable and
	// function names to make it easier to understand. For example, the ReturnSumSq()
	// function takes the place of the FUNK() function (see further comments below).
	// This function is similar to the MaximizeLikelihood function defined previously, the
	// major differences being (a) it can accommodate bigger parameter combinations 
	// (up to 16 parameters), and (b) there are different constraints on the parameters 
	// (this time all parameters have to be positive, but only the first two parameters 
	// have to be <1).

	// VERY IMPORTANT: Always check the FinalSex output file before concluding that
	// you have reached convergence. The sum of squares values might all be very close
	// together, but unless the parameters are also all close to one another, it does not
	// mean that convergence has been reached. Reduce FTol and run the algorithm again,
	// replacing "InitialSex.txt" with "FinalSex.txt", and replacing
	// "FinalSex.txt" with "FinalSex2.txt". (Incidentally, you can use the same
	// method if you reach the maximum number of iterations (500) and wish to continue.)

	int ic, ir, it; // Counters for parameters, vertices and iterations respectively
	int Dimension = 12; // Takes the place of NDIM in AMOEBA
	int Vertices = Dimension + 1; // Takes the place of MPTS in AMOEBA
	double SummedSq[17]; // Takes the place of Y[] in AMOEBA
	double Alpha = 1.0;
	double Beta = 0.5;
	double Gamma = 2.0;
	double ParameterCombinations[17][16]; // Takes the place of matrix P in AMOEBA
	double AveParam[16]; // Takes the place of PBAR in AMOEBA
	double AltParam1[16]; // Takes the place of PR in AMOEBA
	double AltParam2[16]; // Takes the place of PRR in AMOEBA
	double AltSumSq1; // Takes the place of YPR in AMOEBA
	double AltSumSq2; // Takes the place of YPRR in AMOEBA
	int iLowest, iHighest, i2ndHighest; // Take the place of ILO, IHI, INHI
	double RTol;
	int MaxIterations;

	// Determine the starting values of ParameterCombinations and NegLogL arrays
	ReadInitSex(input, ParameterCombinations, Dimension);
	for(ir=0; ir<Vertices; ir++){
		for(ic=0; ic<Dimension; ic++){
			AltParam1[ic] = ParameterCombinations[ir][ic];}
		SummedSq[ir] = ReturnSumSquares(AltParam1);
	}

	// Determine MaxIterations
	if(SexHIVcount<1){
		MaxIterations = 3000;}
	else{
		MaxIterations = 10000;}

	//Determine the starting values of iLowest, iHighest, i2ndHighest, RTol
	iLowest = 0;
	if(SummedSq[0] > SummedSq[1]){
		iHighest = 0;
		i2ndHighest = 1;
	}
	else{
		iHighest = 1;
		i2ndHighest = 0;
	}
	for(ir=0; ir<Vertices; ir++){
		if(SummedSq[ir] < SummedSq[iLowest]){
			iLowest = ir;}
		if(SummedSq[ir] > SummedSq[iHighest]){
			i2ndHighest = iHighest;
			iHighest = ir;
		}
		else if(SummedSq[ir] > SummedSq[i2ndHighest]){
			if(ir!=iHighest){
				i2ndHighest = ir;}
		}
	}
	RTol = 2.0 * fabs(SummedSq[iHighest] - SummedSq[iLowest])/(fabs(SummedSq[iHighest]) +
		fabs(SummedSq[iLowest]));

	// Iterate until convergence is achieved
	it = 0;
	while(RTol > FTol && it < MaxIterations){
		it += 1;
		for(ic=0; ic<Dimension; ic++){ // Calculate average
			AveParam[ic] = 0;}
		for(ir=0; ir<Vertices; ir++){
			if(ir!=iHighest){
				for(ic=0; ic<Dimension; ic++){
					AveParam[ic] += ParameterCombinations[ir][ic];}
			}
		}
		for(ic=0; ic<Dimension; ic++){
			AveParam[ic] = AveParam[ic]/Dimension;
			AltParam1[ic] = (1.0 + Alpha) * AveParam[ic] - Alpha *
				ParameterCombinations[iHighest][ic]; // Reflection step
			if(AltParam1[ic]<0.0){AltParam1[ic]=0.0;}
		}
		if(AltParam1[0]>1.0){AltParam1[0]=1.0;}
		if(AltParam1[1]>1.0){AltParam1[1]=1.0;}
		if(AltParam1[8]<1.0){AltParam1[8]=1.0;}
		if(AltParam1[9]<1.0){AltParam1[9]=1.0;}
		AltSumSq1 = ReturnSumSquares(AltParam1);
		if(AltSumSq1 <= SummedSq[iLowest]){
			for(ic=0; ic<Dimension; ic++){ // Extrapolation step
				AltParam2[ic] = Gamma * AltParam1[ic] + (1.0 - Gamma) * AveParam[ic];
				if(AltParam2[ic]<0.0){AltParam2[ic]=0.0;}
			}
			if(AltParam2[0]>1.0){AltParam2[0]=1.0;}
			if(AltParam2[1]>1.0){AltParam2[1]=1.0;}
			if(AltParam2[8]<1.0){AltParam2[8]=1.0;}
			if(AltParam2[9]<1.0){AltParam2[9]=1.0;}
			AltSumSq2 = ReturnSumSquares(AltParam2);
			if(AltSumSq2 < SummedSq[iLowest]){
				for(ic=0; ic<Dimension; ic++){
					ParameterCombinations[iHighest][ic] = AltParam2[ic];}
				SummedSq[iHighest] = AltSumSq2;
			}
			else{
				for(ic=0; ic<Dimension; ic++){
					ParameterCombinations[iHighest][ic] = AltParam1[ic];}
				SummedSq[iHighest] = AltSumSq1;
			}
		}
		else if(AltSumSq1 >= SummedSq[i2ndHighest]){
			if(AltSumSq1 < SummedSq[iHighest]){
				for(ic=0; ic<Dimension; ic++){
					ParameterCombinations[iHighest][ic] = AltParam1[ic];}
				SummedSq[iHighest] = AltSumSq1;
			}
			for(ic=0; ic<Dimension; ic++){ // Contraction step
				AltParam2[ic] = Beta * ParameterCombinations[iHighest][ic] + (1.0 - Beta) *
					AveParam[ic];
				// Not necessary to check if value is >0
			}
			AltSumSq2 = ReturnSumSquares(AltParam2);
			if(AltSumSq2 < SummedSq[iHighest]){
				for(ic=0; ic<Dimension; ic++){
					ParameterCombinations[iHighest][ic] = AltParam2[ic];}
				SummedSq[iHighest] = AltSumSq2;
			}
			else{
				for(ir=0; ir<Vertices; ir++){
					if(ir!=iLowest){
						for(ic=0; ic<Dimension; ic++){
							AltParam1[ic] = 0.5 * (ParameterCombinations[ir][ic] +
								ParameterCombinations[iLowest][ic]);
							// Not necessary to check if value is in [0, 1] range
							ParameterCombinations[ir][ic] = AltParam1[ic];
						}
						SummedSq[ir] = ReturnSumSquares(AltParam1);
					}
				}
			}
		}
		else{
			for(ic=0; ic<Dimension; ic++){
				ParameterCombinations[iHighest][ic] = AltParam1[ic];}
			SummedSq[iHighest] = AltSumSq1;
		}

		// Lastly, update iLowest, iHighest, i2ndHighest, RTol (same code as before)
		iLowest = 0;
		if(SummedSq[0] > SummedSq[1]){
			iHighest = 0;
			i2ndHighest = 1;
		}
		else{
			iHighest = 1;
			i2ndHighest = 0;
		}
		for(ir=0; ir<Vertices; ir++){
			if(SummedSq[ir] < SummedSq[iLowest]){
				iLowest = ir;}
			if(SummedSq[ir] > SummedSq[iHighest]){
				i2ndHighest = iHighest;
				iHighest = ir;
			}
			else if(SummedSq[ir] > SummedSq[i2ndHighest]){
				if(ir!=iHighest){
					i2ndHighest = ir;}
			}
		}
		RTol = 2.0 * fabs(SummedSq[iHighest] - SummedSq[iLowest])/(fabs(SummedSq[iHighest]) +
			fabs(SummedSq[iLowest]));
	}

	// Generate outputs
	SaveFinalSex(filout, ParameterCombinations, Dimension);

	if(SexHIVcount==0){
		cout.precision(10);
		for(ir=0; ir<Vertices; ir++){
			cout<<"SummedSq["<<ir<<"]: "<<SummedSq[ir]<<endl;}
		for(ic=0; ic<Dimension; ic++){
			cout<<"LS estimate of parameter "<<ic+1<<": "<<ParameterCombinations[iLowest][ic]<<endl;}
		cout<<"Number of iterations: "<<it<<endl;
	}
	if(SexHIVcount>0){
		for(ic=0; ic<Dimension; ic++){
			LSforSex[ic] = ParameterCombinations[iLowest][ic];}
		for(ir=0; ir<Vertices; ir++){
			FinalSumSquares[SexHIVcount-1][ir] = SummedSq[ir];}
	}
}

void GetSingleSexCalib(double Behav[3][2], int AgeStart, int AgeEnd)
{
	int ia;
	double MultPartnerMarriedM, MultPartnerMarriedF, MultPartnerSingleM, MultPartnerSingleF;
	double MarriedM, MarriedF, SingleM, SingleF;
	double NoPartnerM, NoPartnerF;

	MultPartnerMarriedM = 0;
	MultPartnerMarriedF = 0;
	MultPartnerSingleM = 0;
	MultPartnerSingleF = 0;
	MarriedM = 0;
	MarriedF = 0;
	SingleM = 0;
	SingleF = 0;
	NoPartnerM = 0;
	NoPartnerF = 0;
	for(ia=AgeStart; ia<=AgeEnd; ia++){
		MultPartnerMarriedM += MaleHigh.L11.TotalAlive[ia] + MaleHigh.L12.TotalAlive[ia] +
			MaleHigh.L21.TotalAlive[ia] + MaleHigh.L22.TotalAlive[ia];
		MultPartnerMarriedF += FemHigh.L11.TotalAlive[ia] + FemHigh.L12.TotalAlive[ia] +
			FemHigh.L21.TotalAlive[ia] + FemHigh.L22.TotalAlive[ia];
		MultPartnerSingleM += MaleHigh.S11.TotalAlive[ia] + MaleHigh.S12.TotalAlive[ia] +
			MaleHigh.S22.TotalAlive[ia];
		MultPartnerSingleF += FemHigh.S11.TotalAlive[ia] + FemHigh.S12.TotalAlive[ia] +
			FemHigh.S22.TotalAlive[ia];
		MarriedM += MarriedSum[ia][0];
		MarriedF += MarriedSum[ia][1];
		SingleM += TotalPopSum[ia][0] - VirginsSum[ia][0] - MarriedSum[ia][0];
		SingleF += TotalPopSum[ia][1] - VirginsSum[ia][1] - MarriedSum[ia][1];
		NoPartnerM += MaleHigh.NoPartner.TotalAlive[ia] + MaleLow.NoPartner.TotalAlive[ia];
		NoPartnerF += FemHigh.NoPartner.TotalAlive[ia] + FemLow.NoPartner.TotalAlive[ia];
	}

	Behav[0][0] = MultPartnerSingleM/SingleM;
	Behav[0][1] = MultPartnerSingleF/SingleF;
	Behav[1][0] = MultPartnerMarriedM/MarriedM;
	Behav[1][1] = MultPartnerMarriedF/MarriedF;
	Behav[2][0] = NoPartnerM/SingleM;
	Behav[2][1] = NoPartnerF/SingleF;
}

void GetPartnerOutput2005()
{
	// Calculates proportions of individuals with different numbers of partners, for
	// comparison with the sexual behaviour data from the 2005 HSRC household survey

	int ia, ib, ig;

	double Age15to24[3][2];
	double Age25to34[3][2];
	double Age35to44[3][2];
	double Age45to59[3][2];
	double Age60plus[3][2];

	GetSingleSexCalib(Age15to24, 1, 2);
	GetSingleSexCalib(Age25to34, 3, 4);
	GetSingleSexCalib(Age35to44, 5, 6);
	GetSingleSexCalib(Age45to59, 7, 9);
	GetSingleSexCalib(Age60plus, 10, 15);

	for(ig=0; ig<2; ig++){
		UnmarriedMultPartners[0][ig] = Age15to24[0][ig];
		UnmarriedMultPartners[1][ig] = Age25to34[0][ig];
		UnmarriedMultPartners[2][ig] = Age35to44[0][ig];
		UnmarriedMultPartners[3][ig] = Age45to59[0][ig];
		UnmarriedMultPartners[4][ig] = Age60plus[0][ig];
		MarriedMultPartners[0][ig] = Age15to24[1][ig];
		MarriedMultPartners[1][ig] = Age25to34[1][ig];
		MarriedMultPartners[2][ig] = Age35to44[1][ig];
		MarriedMultPartners[3][ig] = Age45to59[1][ig];
		MarriedMultPartners[4][ig] = Age60plus[1][ig];
		UnmarriedSingle[0][ig] = Age15to24[2][ig];
		UnmarriedSingle[1][ig] = Age25to34[2][ig];
		UnmarriedSingle[2][ig] = Age35to44[2][ig];
		UnmarriedSingle[3][ig] = Age45to59[2][ig];
		UnmarriedSingle[4][ig] = Age60plus[2][ig];
	}

	for(ia=0; ia<2; ia++){
		for(ig=0; ig<2; ig++){
			VirginPropn[ia][ig] = VirginsSum[ia+1][ig]/TotalPopSum[ia+1][ig];}
	}

	// Calculate the ConcurrencyBias and AbstinenceBias parameters (new to TSHISA 9)
	for(ig=0; ig<2; ig++){
		ConcurrencyBias[0][ig] = 0.0;
		ConcurrencyBias[1][ig] = 0.0;
		AbstinenceBias[ig] = 0.0;
		for(ia=0; ia<5; ia++){
			if(ig==0 || ia<4){
				ConcurrencyBias[0][ig] += log(UnmarriedMultPartners[ia][ig]/(1.0 - 
					UnmarriedMultPartners[ia][ig])) - log(UnmarriedMultPartnersC[ia][ig]/
					(1.0 - UnmarriedMultPartnersC[ia][ig]));
				ConcurrencyBias[1][ig] += log(MarriedMultPartners[ia][ig]/(1.0 - 
					MarriedMultPartners[ia][ig])) - log(MarriedMultPartnersC[ia][ig]/
					(1.0 - MarriedMultPartnersC[ia][ig]));
			}
			AbstinenceBias[ig] += log(UnmarriedSingle[ia][ig]/(1.0 - 
				UnmarriedSingle[ia][ig])) - log(UnmarriedSingleC[ia][ig]/(1.0 - 
				UnmarriedSingleC[ia][ig]));
		}
		AbstinenceBias[ig] = exp(AbstinenceBias[ig]/5.0);
	}
	ConcurrencyBias[0][0] = exp(ConcurrencyBias[0][0]/5.0);
	ConcurrencyBias[1][0] = exp(ConcurrencyBias[1][0]/5.0);
	ConcurrencyBias[0][1] = exp(ConcurrencyBias[0][1]/4.0);
	ConcurrencyBias[1][1] = exp(ConcurrencyBias[1][1]/4.0);

	/*if(ConcurrencyBias[0][0]<0.5){ConcurrencyBias[0][0] = 0.5;}
	if(ConcurrencyBias[0][0]>2.5){ConcurrencyBias[0][0] = 2.5;}
	if(ConcurrencyBias[0][1]<1.0){ConcurrencyBias[0][1] = 1.0;}
	if(ConcurrencyBias[0][1]>5.0){ConcurrencyBias[0][1] = 5.0;}*/
	if(ConcurrencyBias[1][0]<ConcurrencyBias[0][0]){ErrorInd = 1;}
	//if(ConcurrencyBias[1][0]>5.0){ConcurrencyBias[1][0] = 5.0;}
	if(ConcurrencyBias[1][1]<ConcurrencyBias[0][1]){ErrorInd = 1;}
	if(ConcurrencyBias[0][0]>ConcurrencyBias[0][1]){ErrorInd = 1;}
	if(ConcurrencyBias[1][0]>ConcurrencyBias[1][1]){ErrorInd = 1;}
	/*if(ConcurrencyBias[1][1]>10.0){ConcurrencyBias[1][1] = 10.0;}
	if(AbstinenceBias[1]<0.5){AbstinenceBias[1] = 0.5;}
	if(AbstinenceBias[1]>2.0){AbstinenceBias[1] = 2.0;}*/
	if(AbstinenceBias[0]<AbstinenceBias[1]){ErrorInd = 1;}
	//if(AbstinenceBias[0]>10.0){AbstinenceBias[0] = 10.0;}

	// Make adjustments to true proportions to get reported proportions
	for(ig=0; ig<2; ig++){
		for(ia=0; ia<5; ia++){
			UnmarriedMultPartners[ia][ig] = 1.0/(ConcurrencyBias[0][ig] * (1.0/
				UnmarriedMultPartners[ia][ig] - 1.0) + 1.0);
			MarriedMultPartners[ia][ig] = 1.0/(ConcurrencyBias[1][ig] * (1.0/
				MarriedMultPartners[ia][ig] - 1.0) + 1.0);
			UnmarriedSingle[ia][ig] = 1.0/(AbstinenceBias[ig] * (1.0/
				UnmarriedSingle[ia][ig] - 1.0) + 1.0);
		}
		/*for(ia=0; ia<2; ia++){
			VirginPropn[ia][ig] = 1.0/((1.0/VirginPropn[ia][ig] - 1.0)/DebutBias[ia][ig] + 
				1.0);
		}*/
	}

	// Record outputs in OutPartnerCalib object
	/*if(FixedUncertainty==1){
		for(ig=0; ig<2; ig++){
			for(ia=0; ia<5; ia++){
				OutPartnerCalib.out[CurrSim-1][ia+ig*5] = UnmarriedMultPartners[ia][ig];
				OutPartnerCalib.out[CurrSim-1][10+ia+ig*5] = MarriedMultPartners[ia][ig];
				OutPartnerCalib.out[CurrSim-1][20+ia+ig*5] = UnmarriedSingle[ia][ig];
			}
		}
	}*/
}

double ReturnMarriedPropn(int Sex, int AgeStart, int AgeEnd)
{
	int ia;
	double Married, TotalAlive, MarriedPropn;

	Married = 0;
	TotalAlive = 0;

	for(ia=AgeStart; ia<=AgeEnd; ia++){
		Married += MarriedSum[ia][Sex];
		TotalAlive += TotalPopSum[ia][Sex];
	}

	MarriedPropn = Married/TotalAlive;
	return MarriedPropn;
}

void GetMarriageOutput1996()
{
	// Calculates the proportions of individuals who are married, according to the age bands
	// used in the 1996 census.

	int ia, ig;

	for(ig=0; ig<2; ig++){
		for(ia=0; ia<15; ia++){
			MarriedPropn96[ia][ig] = ReturnMarriedPropn(ig, ia+1, ia+1);}
	}
	/*if(FixedUncertainty==1){
		for(ig=0; ig<2; ig++){
			for(ia=0; ia<15; ia++){
				OutMarried1996.out[CurrSim-1][ia+ig*15] = MarriedPropn96[ia][ig];}
		}
	}*/
}

void GetMarriageOutput2001()
{
	// Calculates the proportions of individuals who are married, according to the age bands
	// used in the 2001 census.

	int ia, ig;

	for(ig=0; ig<2; ig++){
		for(ia=0; ia<15; ia++){
			MarriedPropn01[ia][ig] = ReturnMarriedPropn(ig, ia+1, ia+1);}
	}
	/*if(FixedUncertainty==1){
		for(ig=0; ig<2; ig++){
			for(ia=0; ia<15; ia++){
				OutMarried2001.out[CurrSim-1][ia+ig*15] = MarriedPropn01[ia][ig];}
		}
	}*/
}

void GetMarriageOutput2007()
{
	// Calculates the proportions of individuals who are married, according to the age bands
	// used in the 2007 community survey.

	int ia, ig;

	for(ig=0; ig<2; ig++){
		for(ia=0; ia<15; ia++){
			MarriedPropn07[ia][ig] = ReturnMarriedPropn(ig, ia+1, ia+1);}
	}
	/*if(FixedUncertainty==1){
		for(ig=0; ig<2; ig++){
			for(ia=0; ia<15; ia++){
				OutMarried2007.out[CurrSim-1][ia+ig*15] = MarriedPropn07[ia][ig];}
		}
	}*/
}

void SaveSexCalibOutput(const char* filout)
{
	int ia;
	ofstream file(filout);

	file<<"Propn of men who are virgins (15-19, 20-24)"<<endl;
	file<<VirginPropn[0][0]<<endl;
	file<<VirginPropn[1][0]<<endl;
	file<<"Propn of women who are virgins (15-19, 20-24)"<<endl;
	file<<VirginPropn[0][1]<<endl;
	file<<VirginPropn[1][1]<<endl;
	file<<"Propn of unmarried men with >1 partner"<<endl;
	for(ia=0; ia<5; ia++){
		file<<setw(10)<<UnmarriedMultPartners[ia][0]<<endl;
	}
	file<<"Propn of unmarried women with >1 partner"<<endl;
	for(ia=0; ia<5; ia++){
		file<<setw(10)<<UnmarriedMultPartners[ia][1]<<endl;
	}
	file<<"Propn of married men with >1 partner"<<endl;
	for(ia=0; ia<5; ia++){
		file<<setw(10)<<MarriedMultPartners[ia][0]<<endl;
	}
	file<<"Propn of married women with >1 partner"<<endl;
	for(ia=0; ia<5; ia++){
		file<<setw(10)<<MarriedMultPartners[ia][1]<<endl;
	}
	file<<"Propn of unmarried men with no partner"<<endl;
	for(ia=0; ia<5; ia++){
		file<<setw(10)<<UnmarriedSingle[ia][0]<<endl;
	}
	file<<"Propn of unmarried women with no partner"<<endl;
	for(ia=0; ia<5; ia++){
		file<<setw(10)<<UnmarriedSingle[ia][1]<<endl;
	}
	file<<"Propn of men who are married in 1996"<<endl;
	for(ia=0; ia<15; ia++){
		file<<setw(10)<<MarriedPropn96[ia][0]<<endl;}
	file<<"Propn of women who are married in 1996"<<endl;
	for(ia=0; ia<15; ia++){
		file<<setw(10)<<MarriedPropn96[ia][1]<<endl;}
	file<<"Propn of men who are married in 2001"<<endl;
	for(ia=0; ia<15; ia++){
		file<<setw(10)<<MarriedPropn01[ia][0]<<endl;}
	file<<"Propn of women who are married in 2001"<<endl;
	for(ia=0; ia<15; ia++){
		file<<setw(10)<<MarriedPropn01[ia][1]<<endl;}
	file<<"Propn of men who are married in 2007"<<endl;
	for(ia=0; ia<15; ia++){
		file<<setw(10)<<MarriedPropn07[ia][0]<<endl;}
	file<<"Propn of women who are married in 2007"<<endl;
	for(ia=0; ia<15; ia++){
		file<<setw(10)<<MarriedPropn07[ia][1]<<endl;}
	file.close();
}

void FitHIVandSexData()
{
	int ir, ic;
	ofstream file("GoodnessOfFit.txt");

	ReadRandomAdj();

	// First iteration
	SexHIVcount = 1;
	SexCalib = 1;
	HIVcalib = 0;
	HIVind = 0;
	MinimizeSumSquares(0.0000001, "InitialSex.txt", "FinalSex.txt");
	HIVind = 1;
	HIVcalib = 1;
	SexCalib = 0;
	MaximizeLikelihood(0.0000001, "InitialSimplex.txt", "FinalSimplex.txt");

	// Second iteration
	SexHIVcount = 2;
	SexCalib = 1;
	HIVcalib = 0;
	MinimizeSumSquares(0.0000001, "FinalSex.txt", "FinalSex2.txt");
	HIVcalib = 1;
	MaximizeLikelihood(0.0000001, "FinalSimplex.txt", "FinalSimplex2.txt");
	
	// One could repeat the second step several times to check for convergence. It might 
	// be necessary to increase the number of rows in the FinalLogL, FinalHIVlogL and 
	// FinalSumSquares arrays.

	// Save FinalLogL and FinalSumSquares
	file<<"FinalLogL:"<<endl;
	for(ir=0; ir<5; ir++){
		for(ic=0; ic<8; ic++){
			file<<"	"<<setw(15)<<setprecision(10)<<right<<FinalLogL[ir][ic];}
		file<<endl;
	}
	file<<"FinalHIVlogL:"<<endl;
	for(ir=0; ir<5; ir++){
		for(ic=0; ic<8; ic++){
			file<<"	"<<setw(15)<<setprecision(10)<<right<<FinalHIVlogL[ir][ic];}
		file<<endl;
	}
	file<<"FinalSumSquares:"<<endl;
	for(ir=0; ir<5; ir++){
		for(ic=0; ic<13; ic++){
			file<<"	"<<setw(15)<<setprecision(12)<<right<<FinalSumSquares[ir][ic];}
		file<<endl;
	}
	file.close();
}
