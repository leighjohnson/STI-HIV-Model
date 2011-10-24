#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
// #using <mscorlib.dll>

using namespace std;

// ---------------------------------------------------------------------------------------
// General parameters and settings
// ---------------------------------------------------------------------------------------

int StartYear = 1985;
int CurrYear;
int ProjectionTerm = 26;

int CofactorType = 0; // Nature of STD cofactors in HIV transmission: 0 = no cofactors,
					  // 1 = multiplicative cofactors; 2 = saturation cofactors
int CalculatePAFs = 0; // 1 if you want to calculate PAFs of STDs for HIV transmission,
					   // 0 otherwise
int SyphilisImmunity = 1; // 1 if you want to allow for resistance to syphilis reinfection
						  // when recovered but still seropositive, 0 otherwise
int KeepCurrYearFixed = 0; // 1 if you want to keep the current year fixed at the start year,
						   // 0 otherwise
int NoViralTransm22 = 0; // 1 if you want to assume no transmission of viral STDs in long-
						 // term mutually monogamous relationships, 0 otherwise
int NoViralTransm12 = 0; // 1 if you want to assume no transmission of viral STDs in long-
						 // term relationships if infected partner is monogamous, 0 otherwise
int NoBacterialTransm22 = 1; // 1 if you want to assume no transmission of non-viral STDs in
							 // long-term mutually monogamous relationships, 0 otherwise
int NoBacterialTransm12 = 0; // 1 if you want to assume no transmission of non-viral STDs in
							 // long-term relationships if infected partner is monogamous
int StoreHIVcalib = 0; // 1 if you want to create an output file for the calibration to the
					   // HIV prevalence data, 0 otherwise
int StoreSexCalib = 0; // 1 if you want to create an output file for the calibration to the
					   // sexual behaviour data, 0 otherwise

// Change the following indicators to specify which STDs you want to model (1 = include,
// 0 = exclude):
int HIVind = 1;
int HSVind = 0;
int TPind = 0;
int HDind = 0;
int NGind = 0;
int CTind = 0;
int TVind = 0;
int BVind = 0;
int VCind = 0;

// Change the following indicators to specify which STD prevalence data sets you want to
// include when calculating the likelihood (1 = include, 0 = exclude). This also determines
// which STDs you are considering in the uncertainty analysis.
int HIVcalib = 1;
int HSVcalib = 0;
int TPcalib = 0;
int HDcalib = 0;
int NGcalib = 0;
int CTcalib = 0;
int TVcalib = 0;
int BVcalib = 0;
int VCcalib = 0;
int SexCalib = 1;
int SexHIVcount = 0; // Set to 1 if you wish to calibrate to HIV and sexual behaviour data
					 // simultaneously (the count gets updated with each successive iteration)
int GetSDfromData = 1; // Set to 1 if you want to calculate the standard deviation in the
					   // likelihood function for the national prevalence data based on the
					   // model fit to the data rather than the 95% CIs around the survey
					   // estimates (similar to the approach in Sevcikova et al (2006)).
int GetSpFromData = 0; // Set to 1 if you want to calculate the specificity in the likelihood
					   // function for the national prevalence data based on the model fit
					   // to the data rather than the assumed Sp levels (similar to the bias
					   // adjustment method used by Sevcikova at al (2006)).

int uncertainty = 0; // Set to 1 if you want to conduct uncertainty analysis
int FixedUncertainty = 1; // Set to 1 if you want to do uncertainty analysis, but want to
						  // use previously generated parameter combinations
const int simulations = 1000; // Number of randomly generated parameter combinations to be
							  // evaluated in the uncertainty analysis
const int samplesize = 1000; // Number of parameter combinations to be sampled from initial set

int CycleS = 12; // Number of sexual behaviour cycles per year
int CycleD = 12; // Number of STD cycles per year (NB: must be integer multiple of CycleS)

// ----------------------------------------------------------------------------------------
// Sexual behaviour parameters and arrays
// ----------------------------------------------------------------------------------------

double HighPropnM, HighPropnF; // % of males and females with propensity for >1 partner
double HighPropn15to24[2]; // % of sexually experienced 15-24 year olds with propensity for
						   // >1 partner, by sex
double AssortativeM, AssortativeF; // Degree of assortative mixing in males and females
double GenderEquality; // Gender equality factor
double AnnNumberClients; // Average annual number of sex acts FSW has with clients
double SexualDebut[16][2]; // Continuous rate at which individuals in high risk group start
						   // sexual activity (by age and sex)
double DebutAdjLow[2]; // Factor by which the rate of sexual debut in the high risk group is
					   // multiplied in order to get the rate of debut in the low risk group
double PartnershipFormation[2][2]; // Average annual number of new partners, by risk (first 
								   // index) & sex (2nd index)
double BasePartnerAcqH[2]; // Rate at which a single 15-19 yr old in high risk group acquires
						   // next sexual partner, by sex (base for other calcs)
double AgeEffectPartners[16][2]; // Factor by which average rate of partnership formation
								 // is multiplied to get age-specific rates (by age & sex)
double GammaMeanST[2]; // Mean of gamma density used to determine AgeEffectPartners
double GammaStdDevST[2]; // Std deviation of gamma density used to determine AgeEffectPartners
double PartnerEffectNew[2][2]; // Factor by which average rate of partnership formation is
							   // multiplied to get prob of acquiring ADDITIONAL partner
							   // (1st index is type of current rel, 2nd index is sex)
double HIVeffectPartners[5]; // Factor by which average rate of partnership formation is
							 // multiplied to get rates by HIV stage
double MarriageIncidence[16][2]; // Continuous rate at which marriages are formed, by age &
								 // sex (note: this is a departure from the Excel model)
double MarriageRate[2][4]; // Rate at which ST relationships become marriages, by risk group
						   // of partner and sex/risk of individual (MH, ML, FH, FL)
double AgeEffectMarriage[16][4]; // Factor by which average rate of marriage is multiplied
								 // to get age-specific rates (by age & sex/risk)
double FSWcontactConstant; // Constant term used in determining rate at which men in high 
						   // risk group have contact with FSWs
double MeanFSWcontacts; // Ave annual # sex acts with FSWs, per sexually experienced male
double AgeEffectFSWcontact[16]; // Factor by which average rate of FSW contact is multiplied
								// to get age-specific rates
double GammaMeanFSW; // Mean of gamma density used to determine AgeEffectFSWcontact 
double GammaStdDevFSW; // Std deviation of gamma density used to determine AgeEffectFSWcontact
double PartnerEffectFSWcontact[5]; // Factor by which average rate of FSW contact is 
								   // multiplied to take account of current partners (0=>
								   // no partner, 1=>1ST, 2=>1LT, 3=>2ST, 4=>1ST & 1LT)
double InitFSWageDbn[16]; // Initial proportion of sex workers at each age
double FSWentry[16]; // Relative rates of entry into FSW group, by age
double FSWexit[16]; // Rates of exit from the FSW group, by age
double HIVeffectFSWentry[5]; // Effect of HIV stage on rate of entry into FSW group
double HIVeffectFSWexit[5]; // Effect of HIV stage on rate of exit from FSW group
double MeanDurSTrel[2][2]; // Mean duration (in years) of short-term relationships, by male
						   // risk group (1st index) & female risk group (2nd index)
double LTseparation[16][2]; // Annual rate (cts) at which long-term unions are dissolved, by
							// age and sex
double AgePrefF[16][16]; // Proportion of male partners in each age group, for women of each
						 // age (1st index: female age, 2nd index: male age)
double AgePrefM[16][16]; // Proportion of female partners in each age group, for men of each
						 // age (1st index: male age, 2nd index: female age)
double FreqSexST[16][2]; // Ave # sex acts per ST relationship, per sexual behaviour cycle
						 // (by age and sex)
double FreqSexLT[16][2]; // Ave # sex acts per LT relationship, per sexual behaviour cycle
						 // (by age and sex)
double BaselineCondomUse; // % of sex acts protected among 15-19 females in ST rels in 1998
double BaselineCondomSvy; // As above, but before applying bias adjustment
double RelEffectCondom[3]; // Effect of partnership type on odds of condom use at baseline
double AgeEffectCondom[3]; // Effect of age on odds of condom use, by partnership type 
double RatioInitialTo1998[3]; // Ratio of inital odds of condom use to odds in 1998
double RatioUltTo1998[3]; // Ratio of ultimate odds of condom use to odds in 1998
double MedianToBehavChange[3]; // Median time (in years since 1985) to condom behav change
double ShapeBehavChange[3]; // Weibull shape parameter determining speed of behaviour change
double CondomUseST[16][2]; // Propn of sex acts that are protected in ST rels, by age and sex
double CondomUseLT[16][2]; // Propn of sex acts that are protected in LT rels, by age and sex
double CondomUseFSW; // Propn of sex acts that are protected in FSW-client relationships
double DebutBias[2][2]; // OR relating true odds of sexual experience to reported sexual
						// sexual experience, by age (1st index) and sex (2nd index)
double AbstinenceBias[2]; // OR relating true odds of sexual abstinence to reported sexual
						  // abstinence, by sex
double ConcurrencyBias[2][2]; // OR relating true odds of concurrency to reported concurrency,
							  // by marital status (1st index (0 is single)) & sex (2nd index)
double BehavBiasVar[3][2]; // Sample variance of bias estimates (on logit scale), by type of
						   // behav (1st index) & sex (2nd index)
double CondomScaling; // Parameter to allow for bias in reporting of condoms (1 = no bias)

// ----------------------------------------------------------------------------------------
// Arrays for balancing male and female sexual activity
// ----------------------------------------------------------------------------------------

double DesiredSTpartners[2][2]; // Desired number of new partners, by risk group (1st index)
								// and sex (2nd index)
double DesiredPartnerRiskM[2][2]; // Desired proportion of partners in high and low risk
								  // groups, by male risk group
double DesiredPartnerRiskF[2][2]; // Desired proportion of partners in high and low risk
								  // groups, by female risk group
double AdjSTrateM[2][2]; // Adjustment to rate at which males form partnerships with females
double AdjSTrateF[2][2]; // Adjustment to rate at which females form partnerships with males
double DesiredMarriagesM[2][2]; // Number of new marriages desired by males
double DesiredMarriagesF[2][2]; // Number of new marriages desired by females
double AdjLTrateM[2][2]; // Adjustment to rate at which males marry females
double AdjLTrateF[2][2]; // Adjustment to rate at which females marry males
double ActualPropnLTH[2][2]; // Proportion of long-term partners who are in the high-risk
							 // group, by risk group (1st index) and sex (2nd index)
double DesiredFSWcontacts; // Total numbers of contacts with sex workers (per annum) by men
						   // in the high risk group
double RequiredNewFSW; // Number of women becoming sex workers in current behaviour cycle,
					   // in order to meet excess male demand

// ----------------------------------------------------------------------------------------
// Demographic parameters and arrays
// ----------------------------------------------------------------------------------------

double AgeExitRateM[16][6]; // Rates at which men move out of quinquennial age band, by
							// age and HIV disease stage
double AgeExitRateF[16][6]; // Rates at which women move out of quinquennial age band
double VirginAgeExitRate[16][2]; // Rates at which virgins move out of quinquennial age band
double NonAIDSmortForce[16][2]; // Non-AIDS force of mortality in current year, by age & sex
double NonAIDSmortProb[16][2]; // Prob of non-AIDS death in current cycle, by age & sex
double NonAIDSmortPartner[16][2]; // Force of Non-AIDS mortality among partners, by age & sex
double AIDSmortPartnerM[16][8]; // Force of AIDS mort among fem partners, by age & partnership
double AIDSmortPartnerF[16][8]; // Force of AIDS mort among male partners
double AIDSmortForceM[16][8]; // Force of male AIDS mortality, by age & partnership type
double AIDSmortForceF[16][8]; // Force of female AIDS mortality, by age & partnership type
double HIVnegFert[7]; // Fertility rates in HIV-negative women
double SexuallyExpFert[7]; // Fertility rates in sexually experienced women
double FertilityTable[7][41]; // Fertility rates in HIV-negative women, by age and year
double InfantMort1st6mM[41]; // Prob of death in 1st 6 months of life (males), by year
double ChildMortM[15][41]; // Non-AIDS mortality rates in male children by age and year
double InfantMort1st6mF[41]; // Prob of death in 1st 6 months of life (females), by year
double ChildMortF[15][41]; // Non-AIDS mortality rates in female children by age and year
double NonAIDSmortM[16][41]; // Non-AIDS mortality rates in males by age and year
double NonAIDSmortF[16][41]; // Non-AIDS mortality rates in females by age and year
double StartPop[81][2]; // Numbers of males and females at each individual age (10, ..., 90)
						// as at the start of the projection

// ----------------------------------------------------------------------------------------
// STD parameters not defined in the classes below
// ----------------------------------------------------------------------------------------

double HSVsheddingIncrease[5]; // % increase in HSV-2 shedding by HIV stage
double HSVrecurrenceIncrease[5]; // % increase in recurrence rate by HIV stage
double HSVsymptomInfecIncrease; // Multiple by which HSV-2 infectiousness increased
								// when symptomatic
double InfecIncreaseSyndrome[3][2]; // % by which HIV infectiousness increases when
									// experiencing syndrome
double SuscepIncreaseSyndrome[3][2]; // % by which HIV susceptibility increases when
									// experiencing syndrome
double RatioAsympToAveM; // Ratio of asymptomatic HIV transmission prob to average HIV
						 // transmission prob in males
double RatioAsympToAveF; // Ratio of asymptomatic HIV transmission prob to average HIV
						 // transmission prob in females
double MaxHIVprev; // Maximum HIV prevalence in any cohort in current STD cycle
double RelHIVfertility[5]; // Factor by which fertility rate is multiplied in each HIV stage
double PropnInfectedAtBirth; // Propn of children born to HIV+ mothers who are infected at
							 // or before birth
double PropnInfectedAfterBirth; // Propn of children born to HIV+ mothers who are infected 
								// after birth (through breastfeeding)
double MaleRxRate; // Rate at which adults males seek STD treatment
double MaleTeenRxRate; // Rate at which males aged <20 seek STD treatment
double FemRxRate; // Rate at which adults females seek STD treatment
double FemTeenRxRate; // Rate at which females aged <20 seek STD treatment
double FSWRxRate; // Rate at which female sex workers seek STD treatment
double PropnTreatedPublicM; // Propn of male STD cases treated in public health sector
double PropnTreatedPublicF; // Propn of female STD cases treated in public health sector
double PropnTreatedPrivateM; // Propn of male STD cases treated in private health sector
double PropnTreatedPrivateF; // Propn of female STD cases treated in private health sector
double PropnPublicUsingSM[41]; // Propn of providers in public sector using syndromic mngt
double PropnPrivateUsingSM[41]; // Propn of providers in private sector using syndromic mngt
double DrugShortage[41]; // % redn in public sector treatment effectiveness due to drug
						 // shortages, by year
double HAARTaccess[41]; // % of people progressing to AIDS who start HAART, by year
double PMTCTaccess[41]; // % of pregnant women who are offered PMTCT for HIV, by year
double AcceptScreening; // % of pregnant women offered HIV screening who accept
double AcceptNVP; // % of women testing positive who agree to receive nevirapine
double RednNVP; // % reduction in HIV transmission at/before birth if woman receives NVP
double RednFF; // % reduction in HIV transmission after birth if woman receives formula
			   // feeding OR exclusive breasfeeding
double SecondaryRxMult; // Factor by which the rate of treatment seeking is multiplied
						// when experiencing symptoms of secondary syphilis
double SecondaryCureMult; // Factor by which the probability of cure is multiplied
						  // when treated for secondary syphilis
double FSWasympRxRate; // Rate at which female sex workers seek STD treatment when 
					   // asymptomatically infected with an STD
double FSWasympCure; // Prob that treatment for symptomatic STD in FSW cures other 
					 // asymptomatic STDs (expressed as multiple of the prob of cure if the
					 // STD was symptomatic)
double InitHIVprevHigh; // % of high risk group initially HIV-positive (assumed to be asymp)
double InitHIVtransm[3][2]; // HIV transm probs per sex act at start of epidemic, by
							// nature of rel (1st index) and sex (2nd index)
double RatioUltToInitHIVtransm; // Ratio of ultimate HIV transmission prob to initial HIV
								// transm prob, in the 'no STD cofactor' scenario

// ---------------------------------------------------------------------------------------
// Output values and arrays + calibration statistics
// ---------------------------------------------------------------------------------------

double TotalPopSum[16][2]; // Total number of people in population, by age and sex
double VirginsSum[16][2]; // Total virgins in population, by age and sex
double MarriedSum[16][2]; // Total number of married people in population, by age and sex
double UnmarriedActiveSum[16][2]; // Total unmarried individuals in non-marital relationships
double UnmarriedMultSum[16][2]; // Total unmarried individuals with multiple partners
double MultPartnerSum[16][2]; // Total number of people with >1 partner, by age and sex
double LowRiskSum[16][2]; // Total number of people in low risk group, by age and sex
double HIVstageSumM[16][6]; // Total number of men in each HIV stage, by age
double HIVstageSumF[16][6]; // Total number of women in each HIV stage, by age
double NewHIVsum[16][2]; // Total new HIV infections over last 12 months, by age and sex
/*double NewHIVsumMS[16][2];
double NewHIVsumML[16][2];
double NewHIVsumUS[16][2];
double NewHIVsumCS[16][2];*/
/*double NewHIVsumM[16][2];
double NewHIVsumU[16][2];*/
double NewAIDSdeaths[16][2]; // Total AIDS deaths over last 12 months, by age and sex

// Note that the other output arrays in the 'STD profile' sheet are defined as members of
// objects in the NonHIVtransition class.

double BirthsToHIVmothers[7]; // Births to women with HIV, by age
double TotBirthsToHIVmothers; // Total births to HIV-positive women
double BirthsByAge[7]; // Births (clear and infected), by age
double TotalBirths; // Total births to all women in current year
double NewHIVperinatal; // New HIV infections occurring at or before birth
double NewHIVbreastmilk; // New HIV infections occurring as a result of breastfeeding

// Time series outputs
double HIVprev15to49M[41]; // Male HIV prevalence in the 15-49 age group
double HIVprev15to49F[41]; // Female HIV prevalence in the 15-49 age group
double HIVprevFSW[41]; // HIV prevalence in sex workers
double NGprev15to49M[41]; // Male NG prevalence in the 15-49 age group
double NGprev15to49F[41]; // Female NG prevalence in the 15-49 age group
double NGprevFSW[41]; // NG prevalence in sex workers
double NGprevANC[41]; // NG prevalence in antenatal clinics
double NGprevFPC[41]; // NG prevalence in family planning clinics
double PropnActsProtectedM[41]; // % of sex acts that are protected, among men aged 15-24
double PropnActsProtectedF[41]; // % of sex acts that are protected, among women aged 15-24

// Sexual behaviour outputs and calibration targets
double UnmarriedMultPartners[5][2]; // Propn of unmarried individuals with >1 partner, by age & sex
double MarriedMultPartners[5][2]; // Propn of married individuals with >1 partner, by age & sex
double UnmarriedSingle[5][2]; // Propn of unmarried individuals with no partner, by age & sex
double VirginPropn[2][2]; // Propn of individuals who are sexually inexperienced, by age & sex
double SexuallyExpPropnH[2][2]; // Propn of high risk youth who are sexually experienced,
								// by age and sex
double SexuallyExpPropnL[2][2]; // Propn of low risk youth who are sexually experienced,
								// by age and sex
double MarriedPropn96[15][2]; // Propn of individuals who are married in 1996, by age & sex
double MarriedPropn01[15][2]; // Propn of individuals who are married in 2001, by age & sex
double MarriedPropn07[15][2]; // Propn of individuals who are married in 2007, by age & sex
double UnmarriedMultPartnersC[5][2]; // As above, but representing Calibration targets
double MarriedMultPartnersC[5][2]; 
double UnmarriedSingleC[5][2]; 
double VirginPropnC[2][2]; 
double MarriedPropn95C[5][2]; 
double MarriedPropn98C[7]; 
double MarriedPropn01C[8][2];
double UnmarriedMultPartnersSD[5][2]; // As above, but representing the standard deviations
									  // for the survey proportions
double MarriedMultPartnersSD[5][2]; 
double UnmarriedSingleSD[5][2]; 
double VirginPropnSD[2][2];

// Likelihood and sum of squares statistics
double TotalLogL; // The log of the likelihood for the current simulation (based on HIV and
				  // STD prevalence data)
				  // In some contexts, the likelihood for the sexual behaviour data is added.
double SumSquares; // The log of the likelihood for the current simulation, based on sexual
				   // behaviour data (it's not actually a sum of squares)
double FinalLogL[5][11]; // The negative log likelihood statistics at the end of each 
						 // iteration (1st index), for each vertex in the simplex (2nd index)
double FinalHIVlogL[5][11]; // As before, but relates only to the HIV data
double FinalSumSquares[5][17]; // The sum of squares statistics at the end of each iteration
							   // (1st index), for each vertex in the simplex (2nd index)
double MLEforHIV[10]; // Max likelihood estimates of HIV parameters from previous iteration
double LSforSex[16]; // Least squares estimates of sex parameters from previous iteration
double RandomAdjHIV[8][7]; // Multiplicative adjustments to the final simplex from previous
						   // calibration to HIV prevalence data
double RandomAdjSex[13][12]; // Multiplicative adjustments to the final simplex from previous
							 // calibration to sexual behaviour data

// Uncertainty analysis (SIR)
int CurrSim; // Counter for the parameter combination currently being run
int ErrorInd; // Indicates whether the current parameter combination produces unreasonable
			  // sexual behaviour parameters (ErrorInd=1) or not (ErrorInd=0)
int sampleid[samplesize]; // Stores the ID numbers for the sampled simulations
int ErrorCount; // Counts the cumulative number of parameter combinations for which errors
				// have been generated

// Uncertainty analysis (MCMC)
int CumIterations; // The length of each series generated thus far
const int NumberSeries = 10; // The number of series generated
int RunLength = 20000; // The number of iterations in each series in the current run
const int MCMCdim = 14; // The number of parameters varied in the MCMC algorithm
double JumpingDbnVarAdj; // Factor by which covariance matrix is multiplied to get covariance
						 // of the multivariate normal jumping distribution (c^2 in Gelman)
int TuningPeriod = 2000; // Last iteration at which JumpingDbnVarAdj is updated
double TuningFreq = 25; // Number of iterations between updates of JumpingDbnVarAdj (but note
						// that updates of Covariance over-ride updates of JumpingDbnVarAdj)
int UpdateCovarPeriod = 1400; // Last iteration at which Covariance matrix is updated
double UpdateCovarFreq = 200; // Number of iterations between updates of Covariance matrix
double Covariance[MCMCdim][MCMCdim]; // The covariance matrix for the MCMC parameters
double Cholesky[MCMCdim][MCMCdim]; // The Cholesky decomposition of the covariance matrix
								   // (Note that this matrix is UPPER triangular)
double AcceptanceRate; // Proportion of randomly generated parameter combinations accepted

double FPCweights[16]; // Weights are the proportions of women using modern contraception
double TotalGUDcases[2]; // Total # individuals with genital ulcers, in males & females

// ---------------------------------------------------------------------------------------
// Classes for STD prevalence data and likelihoods
// ---------------------------------------------------------------------------------------

class PrevalenceData
{
public:
	PrevalenceData();

	double LogL;
	int Observations;
	int SexInd; // 0 = males, 1 = females
};

class NationalData: public PrevalenceData
{
public:
	NationalData();

	// Note that in defining the arrays below we are assuming that there would not be more
	// than 100 data points to calibrate to. If for some reason there are (i.e. Observations
	// >100), then one should change the dimensions of the arrays below.

	int StudyYear[100];
	double StudyPrev[100];
	double PrevSE[100];
	int AgeStart[100];
	double ExpSe[100];
	double ExpSp[100];
	double ModelPrev[100];
	double BiasMult[16];

	void CalcLogL();
};

class AntenatalN: public NationalData
{
public:
	AntenatalN();
};

class HouseholdN: public NationalData
{
public:
	HouseholdN();
};

class SentinelData: public PrevalenceData
{
public:
	SentinelData();

	// Note that in defining the arrays below we are assuming that there would not be more
	// than 25 data points to calibrate to. If for some reason there are (i.e. Observations
	// >25), then one should change the dimensions of the arrays below.

	int StudyYear[25];
	int StudyN[25];
	double StudyPrev[25];
	double StudyPos[25];
	double ExpSe[25];
	double ExpSp[25];
	double VarSe[25];
	double VarSp[25];
	int HIVprevInd[25]; // 1 if HIV prevalence was measured in the study, 0 otherwise
	double HIVprev[25];
	double ModelPrev[25];
	double BiasMult;
	double VarStudyEffect; // Sigma(b) squared

	void CalcLogL();
};

class Household: public SentinelData
{
public:
	Household();

	int AgeStart[25];
	int AgeEnd[25];
};

class NonHousehold: public SentinelData
{
public:
	NonHousehold();
};

class ANC: public NonHousehold
{
public:
	ANC();
};

class FPC: public NonHousehold
{
public:
	FPC();
};

class GUD: public NonHousehold
{
public:
	GUD();
};

class CSW: public NonHousehold
{
public:
	CSW();
};

// ---------------------------------------------------------------------------------------
// Classes for outputs
// ---------------------------------------------------------------------------------------

class OutputArray
{
public:
	OutputArray(int n);

	int columns;
	double out[simulations][20]; // None of the arrays require > 20 columns. 

	void Record(const char* filout, int n);
	void RecordSample(const char* filout, int n);
	void SampleInput();
};

class MCMCparameter: public OutputArray
{
public:
	MCMCparameter(int n): OutputArray(n){columns = n;}

	int index;
	double Param1, Param2; // Parameters for the prior distribution
	int PriorType; // 1 = beta, 2 = gamma, 3 = uniform
	double mean;
	double BetweenVar, WithinVar, ScaleRedn;
	double SeriesMean[NumberSeries];
	double SeriesVar[NumberSeries];

	double GetLogPrior(double x);
	void GetMean(int n);
	void GetCovar(MCMCparameter* a, int n);
	void GetAllCovar(int n);
	void TestConvergence(double pi); // pi is the propn of CumIterations included
	void ReadCumIterations(const char* input);
};

class PostOutputArray
{
	// Same as OutputArray class except that we only use this to record outputs from
	// the posterior distribution (and hence require smaller output array).
public:
	PostOutputArray(int n);

	int columns;
	double out[samplesize][41]; // None of the arrays require > 41 columns. 

	void RecordSample(const char* filout);
};

// ---------------------------------------------------------------------------------------
// Classes for STD parameters and probabilities of transition between STD disease states
// ---------------------------------------------------------------------------------------

class STDtransition
{
public:
	STDtransition();

	// NB: The CondomEff, SuscepIncrease and InfectProb members all actually belong to the
	// opposite sex. This is potentially confusing, but we have done things this way so that
	// the TransmProb and InfectProb arrays can be calculated within the same STDtransition
	// object (the code would be a lot more complicated if we had to calculate the InfectProb
	// in the STDtransition male object from the TransmProb in the STDtransition fem object).

	int nStates;
	int SexInd; // 0 = male, 1 = female
	double AveDuration[6]; // Average number of weeks spent in state if untreated
	double CondomEff; // Probability that condom prevents transmission
	double SuscepIncrease[16]; // Multiple by which susceptibility to STD increases, by age
	double HIVinfecIncrease[6]; // % by which HIV infectiousness increases, by HIV/STD stage
	double RelTransmCSW; // Ratio of M->F transm prob per sex act in CSW-client relationships 
						 // to that in other relationships
	double TransmProbS1to1[16]; // Prob that indiv in high risk group transmits STD to
								// short-term partner in high risk group in single act of sex
	double TransmProbS1to2[16];
	double TransmProbS2to1[16];
	double TransmProbS2to2[16];
	double TransmProbL1to1[16]; // Prob that indiv in high risk group transmits STD to
								// long-term partner in high risk group in single act of sex
	double TransmProbL1to2[16];
	double TransmProbL2to1[16];
	double TransmProbL2to2[16];
	double InfectProbS1from1[16]; // Prob that indiv in high risk group becomes infected
								  // with STD in single act of sex with high-risk ST partner
	double InfectProbS2from1[16];
	double InfectProbS1from2[16];
	double InfectProbS2from2[16];
	double InfectProbL1from1[16]; // Prob that indiv in high risk group becomes infected
								  // with STD in single act of sex with high-risk LT partner
	double InfectProbL2from1[16];
	double InfectProbL1from2[16];
	double InfectProbL2from2[16];
	double InfectProbFSW[16]; // Prob that indiv becomes infected with STD in single act of
							  // sex with FSW (if male) or client (if FSW)

	// Objects used for calibration purposes
	ANC ANClogL;
	FPC FPClogL;
	GUD GUDlogL;
	CSW CSWlogL;
	Household HouseholdLogL;
	AntenatalN AntenatalNlogL;
	HouseholdN HouseholdNlogL;
	double CSWprevalence;

	void ClearTransmProb();
	void CalcTransmProb();
	void CalcInfectProb(double Transm[16], double Infect[16], int RelType);
	void CalcAllInfectProb();
	void ReadPrevData(const char* input);
	void GetCSWprev();
	void SetVarStudyEffect(double Variance);
};

class HIVtransition: public STDtransition
{
public:
	HIVtransition(int Sex, int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, int ObsHH, 
		int ObsANCN, int ObsHHN);

	double TransmProb[7]; // Transmission probability per sex act (by relationship type)
	double From1to2; // Prob of progressing from acute HIV to asymp per STD cycle
	double From2to3; // Prob of progressing from asymp to pre-AIDS symptoms, per STD cycle
	double From3to4; // Prob of progressing from pre-AIDS to untreated AIDS per STD cycle
	double From3to5; // Prob of progressing to AIDS & starting HAART per STD cycle
	double From4toDead; // Prob of dying from AIDS if not receiving HAART
	double From5toDead; // Prob of dying from AIDS while receiving HAART

	void CalcTransitionProbs();
	void GetANCprev();
	void GetHHprev();
};

class NonHIVtransition: public STDtransition
{
public:
	NonHIVtransition();

	double CorrectRxPreSM; // % of cases correctly treated before syndromic management 
	double CorrectRxWithSM; // % of cases correctly treated under syndromic management
	double DrugEff; // Prob of complete cure if effective drugs are prescribed
	double TradnalEff; // Prob of cure if treated by a traditional healer
	double ProbCompleteCure; // Prob that individual seeking treatment is cured
	double HIVsuscepIncrease[6]; // % by which HIV susceptibility increases, by STD stage

	// Output arrays in the 'STD profile' sheet
	double AliveStage0[16][7]; // Total pop alive with no HIV, by age and STD stage
	double AliveStage1[16][7]; // Total pop alive in HIV stage 1, by age and STD stage
	double AliveStage2[16][7];
	double AliveStage3[16][7];
	double AliveStage4[16][7];
	double AliveStage5[16][7];
	double AliveSum[16][7]; // Total pop alive, by age and STD stage

	void CalcProbCure();
	void ClearAlive();
	void CalcTotalAlive();
};

class TradnalSTDtransition: public NonHIVtransition
{
public:
	TradnalSTDtransition();

	double TransmProb; // Probability of transmission per act of sex
	double TransmProbSW; // Probability of transmission per act of sex (note that this is
						 // only used for client-to-sex worker transmission)

	void GetANCprev();
	void GetHHprev();
	void GetFPCprev();
};

class SyphilisTransition: public TradnalSTDtransition
{
public:
	SyphilisTransition(int Sex, int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, int ObsHH, 
		int ObsANCN, int ObsHHN);

	double ANCpropnScreened; // % of women attending ANCs who are screened for syphilis
	double ANCpropnTreated; // % of women testing positive who receive treatment
	double ANCpropnCured; // % of women attending ANCs who get cured
	double ProbANCcured[16]; // Prob that a woman with syphilis attends an ANC and is cured,
							// per STD cycle (by age: 10-14,15-19, ...)

	double From1to2; // Prob that incubating syphilis becomes primary syphilis per STD cycle
	double From2to3; // Prob that primary syphilis becomes secondary syphilis per STD cycle
	double From2to3T; // Prob that primary syphilis becomes secondary syphilis, age <20
	double From2to3C; // Prob that primary syphilis becomes secondary syphilis, in CSWs
	double From2to5; // Prob that primary syphilis is cured per STD cycle
	double From2to5T; // Prob that primary syphilis is cured per STD cycle, age <20
	double From2to5C; // Prob that primary syphilis is cured per STD cycle, in CSWs
	double From3to4; // Prob that secondary syphilis becomes latent per STD cycle
	double From3to4T; // Prob that secondary syphilis becomes latent per STD cycle, age <20
	double From3to4C; // Prob that secondary syphilis becomes latent per STD cycle, in CSWs
	double From3to5; // Prob that secondary syphilis is cured per STD cycle
	double From3to5T; // Prob that secondary syphilis is cured per STD cycle, age <20
	double From3to5C; // Prob that secondary syphilis is cured per STD cycle, in CSWs
	double From4to6; // Prob that latent syphilis resolves per STD cycle
	double From4to6C; // Prob that latent syphilis resolves per STD cycle, in CSWs
	double From5to0; // Prob of seroreversion if cured in early syphilis, per STD cycle
	double From6to0; // Prob of seroreversion if resolved in latent syphilis, per STD cycle

	void CalcTransitionProbs();
	void GetGUDprev();
};

class HerpesTransition: public TradnalSTDtransition
{
public:
	HerpesTransition(int Sex, int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, int ObsHH, 
		int ObsANCN, int ObsHHN);

	double RecurrenceRate; // Rate at which symptomatic recurrences occur
	double SymptomaticPropn; // % of people who develop primary ulcer
	double From1to2; // Prob that primary ulcer resolves per STD cycle
	double From1to2T; // Prob that primary ulcer resolves per STD cycle, age <20
	double From1to2C; // Prob that primary ulcer resolves per STD cycle, in CSWs
	double From2to3[6]; // Prob of symptomatic recurrence per STD cycle, by HIV stage
	double From3to2; // Prob of resolution of recurrent ulcer per STD cycle
	double From3to2T; // Prob of resolution of recurrent ulcer per STD cycle, age <20
	double From3to2C; // Prob of resolution of recurrent ulcer per STD cycle, in CSWs
	double From2to4; // Prob that transiently asymp indiv becomes permanently asymp

	void CalcTransitionProbs();
	void GetGUDprev();
};

class OtherSTDtransition: public TradnalSTDtransition
{
public:
	OtherSTDtransition(int Sex, int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, int ObsHH, 
		int ObsANCN, int ObsHHN);

	double SymptomaticPropn; // % of individuals who develop symptoms
	double From1to0; // Prob of resolution of symptomatic infection per STD cycle
	double From1to0T; // Prob of resolution of symptomatic infection per STD cycle, age <20
	double From1to0C; // Prob of resolution of symptomatic infection per STD cycle, in CSWs
	double From2to0; // Prob of resolution of asymptomatic infection per STD cycle
	double From2to0C; // Prob of resolution of asymptomatic infection per STD cycle, in CSWs

	void CalcTransitionProbs();
	void GetGUDprev();
};

class NonSTDtransition: public NonHIVtransition
{
public:
	NonSTDtransition();

	double DrugPartialEff; // Prob of partial cure if effective drugs are prescribed
	double ProbPartialCure; // Prob that individual seeking treatment is partially cured

	void CalcProbPartialCure();
};

class BVtransition: public NonSTDtransition
{
public:
	BVtransition(int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, int ObsHH, int ObsANCN,
		int ObsHHN);

	double SymptomaticPropn; // Propn of women developing BV who experience symptoms
	double Incidence1; // Weekly incidence of BV in women with intermediate flora, with
					   // one current partner
	double IncidenceMultTwoPartners; // Multiple by which incidence is increased in women
									 // with more than one partner
	double IncidenceMultNoPartners; // Multiple by which incidence is decreased in women
									// with no partners
	double CtsTransition[4][4]; // Continuous rates of transition between states
	double From1to2; // Prob of going from normal vaginal flora to intermediate per STD cycle
	double From2to1ind; // Independent prob of reverting to normal flora per STD cycle
	double From3to1; // Prob of returning to normal flora if BV is currently symptomatic
	double From3to1T; // Prob of returning to normal flora, age <20
	double From3to1C; // Prob of returning to normal flora, in CSWs
	double From3to2; // Prob of returning to intermediate flora if BV is currently symptomatic
	double From3to2T; // Prob of returning to intermediate flora, age <20
	double From3to2C; // Prob of returning to intermediate flora, in CSWs
	double From4to1; // Prob of returning to normal flora if BV is currently asymp
	double From4to1C; // Prob of returning to normal flora if BV is currently asymp, in CSWs
	double From4to2; // Prob of returning to intermediate flora if BV is currently asymp
	double From4to2C; // Prob of returning to intermediate flora if BV is asymp, in CSWs
	double From2to3ind[3]; // Independent prob of developing symptomatic BV per STD cycle, if 
						   // currently intermediate, by # current partners
	double From2to4ind[3]; // Independent prob of developing asymp BV per STD cycle, if 
						   // currently intermediate, by # current partners
	double From2to1dep[3]; // Dependent prob of reverting to normal flora per STD cycle, if 
						   // currently intermediate, by # current partners
	double From2to3dep[3]; // Dependent prob of developing symptomatic BV per STD cycle, if 
						   // currently intermediate, by # current partners
	double From2to4dep[3]; // Dependent prob of developing asymp BV per STD cycle, if 
						   // currently intermediate, by # current partners

	void CalcTransitionProbs();
	void GetANCprev(); 
	void GetFPCprev();
	// Note that we haven't defined a GetHHprev function, since there are no household
	// survey estimates of BV prevalence.
};

class VCtransition: public NonSTDtransition
{
public:
	VCtransition(int ObsANC, int ObsFPC, int ObsGUD, int ObsCSW, int ObsHH, int ObsANCN,
		int ObsHHN);

	double RecurrenceRate; // Rate at which symptoms develop in women with yeast colonization
	double Incidence; // Rate at which asymptomatic yeast colonization occurs
	double IncidenceIncrease[5]; // % increase in incidence of yeasts by HIV stage
	double From1to2; // Prob that asymp colonization becomes symptomatic per STD cycle
	double From1to2C; // Prob that asymp colonization becomes symptomatic, in CSWs
	double From1to0; // Prob that asymp colonization resolves per STD cycle
	double From1to0C; // Prob that asymp colonization resolves per STD cycle, in CSWs
	double From2to1; // Prob that symptomatic infection becomes asymp per STD cycle
	double From2to1T; // Prob that symptomatic infection becomes asymp per STD cycle, age <20
	double From2to1C; // Prob that symptomatic infection becomes asymp per STD cycle, in CSWs
	double From2to0; // Prob that symptomatic infection is cured per STD cycle
	double From2to0T; // Prob that symptomatic infection is cured per STD cycle, age <20
	double From2to0C; // Prob that symptomatic infection is cured per STD cycle, in CSWs
	double From0to1[7][6]; // Prob that uninfected woman gets asymptomatically colonized per
						   // STD cycle (by age and HIV stage)
	
	void CalcTransitionProbs();
	void GetANCprev();
	void GetFPCprev();
	// Note that we haven't defined a GetHHprev function, since there are no household
	// survey estimates of candidiasis prevalence.
};

// ---------------------------------------------------------------------------------------
// Classes for STD objects within each risk group
// ---------------------------------------------------------------------------------------

class STD
{
public:
	STD();
	int nStates;
};

class NonHIV: public STD
{
public:
	NonHIV();

	double PropnByStage0[16][7]; // Propn of individuals in each STD stage (2nd index),
								 // among individuals aged x (1st index) who are HIV-negative
	double PropnByStage1[16][7]; // Propn of individuals in each STD stage (2nd index),
								 // among individuals aged x (1st index) who are HIV stage 1
	double PropnByStage2[16][7]; // Ditto for HIV stage 2 etc
	double PropnByStage3[16][7];
	double PropnByStage4[16][7];
	double PropnByStage5[16][7];
	double TempPropnByStage0[16][7]; // Same as before, used in intermediate calculations
	double TempPropnByStage1[16][7]; // (e.g. calculating transitions between STD stages)
	double TempPropnByStage2[16][7];
	double TempPropnByStage3[16][7];
	double TempPropnByStage4[16][7];
	double TempPropnByStage5[16][7];
	double NumberByStage0[16][7]; // Number of individuals in STD stage z (2nd index), aged
								  // x (1st index), who are HIV-negative
	double NumberByStage1[16][7]; // Number of individuals in STD stage z (2nd index), aged
								  // x (1st index), who are HIV-positive in stage 1
	double NumberByStage2[16][7]; // Ditto for HIV stage 2 etc
	double NumberByStage3[16][7];
	double NumberByStage4[16][7];
	double NumberByStage5[16][7];

	void GetTotalBySTDstage(NonHIVtransition* a);
};

class NonSTD: public NonHIV
{
public:
	NonSTD();
};

class BV: public NonSTD
{
public:
	BV();

	void CalcSTDtransitions(BVtransition* a, int FSWind, int Partners);
	void CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7],
		BVtransition* a, int FSWind, int Partners);
};

class VC: public NonSTD
{
public:
	VC();

	void CalcSTDtransitions(VCtransition* a, int FSWind);
	void CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7],
		VCtransition* a, int FSWind, int is);
};

class TradnalSTD: public NonHIV
{
public: 
	TradnalSTD();

	double WeightedProbTransm[16]; // Probability of transmitting STD per act of unprotected 
								   // sex, multiplied by # infectious individuals
	double WeightedProbTransmFSW; // Probability of transmitting STD per act of unprotected 
								  // sex, multiplied by # infectious men visiting FSWs
	double InfectProb[16]; // Prob of acquiring STD, in a single STD cycle
};

class Syphilis: public TradnalSTD
{
public:
	Syphilis();

	void CalcTransmissionProb(SyphilisTransition* a);
	void CalcSTDtransitions(SyphilisTransition* a, int FSWind);
	void CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7],
		SyphilisTransition* a, int FSWind);
};

class MaleTP: public Syphilis
{
public:
	MaleTP();
};

class FemTP: public Syphilis
{
public:
	FemTP();
};

class Herpes: public TradnalSTD
{
public:
	Herpes();

	void CalcTransmissionProb(HerpesTransition* a);
	void CalcSTDtransitions(HerpesTransition* a, int FSWind);
	void CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7],
		HerpesTransition* a, int FSWind, int is);
};

class MaleHSV: public Herpes
{
public:
	MaleHSV();
};

class FemHSV: public Herpes
{
public:
	FemHSV();
};

class OtherSTD: public TradnalSTD
{
public:
	OtherSTD();

	void CalcTransmissionProb(OtherSTDtransition* a);
	void CalcSTDtransitions(OtherSTDtransition* a, int FSWind);
	void CalcSTDstageTransitions(double Propn[16][7], double TempPropn[16][7],
		OtherSTDtransition* a, int FSWind);
};

class MaleHD: public OtherSTD
{
public:
	MaleHD();
};

class FemHD: public OtherSTD
{
public:
	FemHD();
};

class MaleNG: public OtherSTD
{
public:
	MaleNG();
};

class FemNG: public OtherSTD
{
public:
	FemNG();
};

class MaleCT: public OtherSTD
{
public:
	MaleCT();
};

class FemCT: public OtherSTD
{
public:
	FemCT();
};

class MaleTV: public OtherSTD
{
public:
	MaleTV();
};

class FemTV: public OtherSTD
{
public:
	FemTV();
};

class PaedHIV
{
public:
	PaedHIV();

	double PreAIDSmedian;
	double PreAIDSshape;
	double MeanAIDSsurvival;
	double PreAIDSstage[15];
	double AIDSstage[15];
	double NewHIV;
	double AIDSprogressionRate[15]; // Independent probability of progressing to AIDS, by age
	double AIDSprob1st6m; // Independent prob of progressing to AIDS in 1st 6 months of life
	double NonAIDSmort[15];
	double MortProb1st6m; // Independent prob of non-AIDS mortality in 1st 6 months of life
	double AIDSdeaths[15]; // Deaths due to AIDS, by age at END of year

	void CalcAIDSprogression();
	void CalcAgeChanges();
};

// ----------------------------------------------------------------------------------------
// Sexual behaviour/population group classes
// ----------------------------------------------------------------------------------------

class SexBehavGrp
{
public:
	SexBehavGrp();

	int SexInd; // 0 = male, 1 = female
	int RiskGroup; // 1 = high risk, 2 = low risk (same as in Excel model)
};

class SexuallyExp: public SexBehavGrp
{
public:
	SexuallyExp();

	double NumbersByHIVstage[16][6];
	double TempNumbersByHIVstage[16][6];
	double TotalAlive[16];
	double STDcofactor[16][6];
	double GUDpropn[16][6]; // Propn with GUD, by age (1st index) & HIV stage (2nd index)
	double DischargePropn[16][6]; // Propn with discharge (& no GUD), by age & HIV stage
	double AsympSTDpropn[16][6]; // Propn with asymp STD, by age & HIV stage
	double WeightedProbTransm[16][7]; // Probability of transmitting HIV per act of  
									  // unprotected sex, multiplied by # infectious 
									  // individuals (by age & type of partner)
	double WeightedProbTransmFSW; // Probability of transmitting HIV per act of unprotected 
								  // sex, multiplied by # infectious men visiting FSWs
	double HIVinfectProb[16]; // Prob of being infected with HIV in a single STD cycle
	//double HIVinfectProbST[16];
	//double HIVinfectProbLT[16];
	//double HIVinfectProbCS[16];
	double HIVstageExits[16][6]; // Numbers of exits from each HIV stage over STD cycle

	double DesiredNewPartners; // Desired number of new short-term partners
	double DesiredNewL1; // Desired number of marriages to individuals in the high-risk group
	double DesiredNewL2; // Desired number of marriages to individuals in the low-risk group
	
	double AcquireNewS1[16][6]; // Independent prob of acquiring new ST partner in high risk
								// group over current sexual behaviour cycle
	double AcquireNewS2[16][6];
	double MarryL1[16][6]; // Independent prob of marrying current high-risk ST partner 
	double MarryL2[16][6];
	double LoseL[16][6]; // Independent prob of losing current LT partner over current sexual
						 // behav cycle (due to divorce or death)
	double LoseS1[16][6];
	double LoseS2[16][6];
	double EnterSW[16][6]; // Independent prob of becoming a sex workers over current sexual
						   // behav cycle
	double LeaveSW[16][6]; // Independent prob of ceasing sex work

	// Dependent versions of corresponding independent rates defined above:
	double AcquireNewS1dep[16][6];
	double AcquireNewS2dep[16][6];
	double MarryL1dep[16][6];
	double MarryL2dep[16][6];
	double LoseLdep[16][6];
	double LoseS1dep[16][6];
	double LoseS2dep[16][6];
	double EnterSWdep[16][6];
	double LeaveSWdep[16][6];
	double Remain[16][6];
	
	int NumberPartners;
	int NumberPartnersS1;
	int NumberPartnersS2;
	int NumberPartnersL1;
	int NumberPartnersL2;
	int MarriedInd;
	int FSWind; // Determines whether women are FSWs (0 = no, 1 = yes)
	int VirginInd; // Determines whether individuals are virgins (0 = no, 1 = yes)
	int FSWcontactOffset; // Determines adjustment to rate of FSW contact in high-risk males
	double AnnFSWcontacts; // Annual number of contacts with FSWs

	void SetPartnerNumbers(int S1, int S2, int L1, int L2);
	void GetNumbersBySTDstage(NonHIV* m);
	void GetPropnsBySTDstage(NonHIV* m);
	void GetSTDcofactor(NonHIV* m, NonHIVtransition* a);
	void CalcHIVtransmProb(HIVtransition* a);
	void CalcHIVtransitions(HIVtransition* a);
	void HIVstageChanges(NonHIV* a); // Similar to HIVstageChanges macro in Excel model
	void SetHIVnumbersToTemp();
	void GetNewPartners();
	void GetPartnerTransitions();
	void ConvertToDependent1(double IndArray1[16][6], double DepArray1[16][6]);
	void ConvertToDependent2(double IndArray1[16][6], double IndArray2[16][6], 
		double DepArray1[16][6], double DepArray2[16][6]);
	void ConvertToDependent3(double IndArray1[16][6], double IndArray2[16][6],
		double IndArray3[16][6], double DepArray1[16][6], double DepArray2[16][6], 
		double DepArray3[16][6]);
	void ConvertToDependent4(double IndArray1[16][6], double IndArray2[16][6],
		double IndArray3[16][6], double IndArray4[16][6], double DepArray1[16][6], 
		double DepArray2[16][6], double DepArray3[16][6], double DepArray4[16][6]);
	void CalcAgeChanges(NonHIV* m);
	double ReturnHIVprev();
	double ReturnTotalGUD();
};

class SexuallyExpM: public SexuallyExp
{
public:
	SexuallyExpM();

	MaleHSV MHSV;
	MaleTP MTP;
	MaleHD MHD;
	MaleNG MNG;
	MaleCT MCT;
	MaleTV MTV;

	void Reset();
	void GetAllNumbersBySTDstage();
	void GetAllPropnsBySTDstage();
	void UpdateSyndromePropns();
	void GetAllSTDcofactors();
	void CalcTransmissionProb();
	void CalcOneTransmProb(STDtransition* a, TradnalSTD* b);
	void CalcInfectionProb();
	void CalcOneInfectProb(STDtransition* a, TradnalSTD* b);
	void CalcHSVinfectProb(HerpesTransition* a, Herpes* b);
	void CalcHIVinfectProb(HIVtransition* a);
	void CalcSTDtransitions();
	void AllHIVstageChanges();
	void GetNumbersRemaining1(NonHIV* a);
	void GetNumbersRemaining();
	void CalcAllAgeChanges();
	void GetTotalBySTDstage();
	void RecordPropnsByStage(ofstream* file);
};

class SexuallyExpF: public SexuallyExp
{
public:
	SexuallyExpF();

	FemHSV FHSV;
	FemTP FTP;
	FemHD FHD;
	FemNG FNG;
	FemCT FCT;
	FemTV FTV;
	BV FBV;
	VC FVC;

	void Reset();
	void GetAllNumbersBySTDstage();
	void GetAllPropnsBySTDstage();
	void UpdateSyndromePropns();
	void GetAllSTDcofactors();
	void CalcTransmissionProb();
	void CalcOneTransmProb(STDtransition* a, TradnalSTD* b);
	void CalcInfectionProb();
	void CalcOneInfectProb(STDtransition* a, TradnalSTD* b);
	void CalcHSVinfectProb(HerpesTransition* a, Herpes* b);
	void CalcHIVinfectProb(HIVtransition* a);
	void CalcSTDtransitions();
	void AllHIVstageChanges();
	void GetNumbersRemaining1(NonHIV* a);
	void GetNumbersRemaining();
	void CalcAllAgeChanges();
	void GetTotalBySTDstage();
	void GetCSWprev(STDtransition* a, NonHIV* b, int BVindicator);
	void RecordPropnsByStage(ofstream* file);
};

class SexCohort
{
public:
	SexCohort();

	int SexInd; // 0 = male, 1 = female
	int RiskGroup; // 1 = high risk, 2 = low risk
	double TotalGUDcases;
};

class SexCohortM: public SexCohort
{
public:
	SexCohortM(int Risk);

	SexuallyExpM Virgin; // Note that we are creating this object from the SexuallyExpM
						 // class for convenience, though these men are not sexually
						 // experienced.
	SexuallyExpM NoPartner;
	SexuallyExpM S1;
	SexuallyExpM S2;
	SexuallyExpM L1;
	SexuallyExpM L2;
	SexuallyExpM S11;
	SexuallyExpM S12;
	SexuallyExpM S22;
	SexuallyExpM L11;
	SexuallyExpM L12;
	SexuallyExpM L21;
	SexuallyExpM L22;

	void Reset();
	void GetAllNumbersBySTDstage();
	void GetAllPropnsBySTDstage();
	void UpdateSyndromePropns();
	void GetAllSTDcofactors();
	void CalcTransmissionProb();
	void CalcInfectionProb();
	void CalcSTDtransitions();
	void HIVstageChanges();
	void SetHIVnumbersToTemp();
	void GetNewPartners();
	void GetPartnerTransitions();
	void GetAllNumbersRemaining();
	void GetNumbersChanging(SexuallyExpM* a, SexuallyExpM* b, double RatesDep[16][6]);
	void GetAllNumbersChanging();
	void CalcAllAgeChanges();
	void GetNewVirgins();
	void GetTotalBySTDstage();
	void GetTotalGUD();
};

class SexCohortF: public SexCohort
{
public:
	SexCohortF(int Risk);

	SexuallyExpF Virgin; // Note that we are creating this object from the SexuallyExpF
						 // class for convenience, though these women are not sexually
						 // experienced.
	SexuallyExpF FSW;
	SexuallyExpF NoPartner;
	SexuallyExpF S1;
	SexuallyExpF S2;
	SexuallyExpF L1;
	SexuallyExpF L2;
	SexuallyExpF S11;
	SexuallyExpF S12;
	SexuallyExpF S22;
	SexuallyExpF L11;
	SexuallyExpF L12;
	SexuallyExpF L21;
	SexuallyExpF L22;

	void Reset(); // This function also sets the initial HIV prevalence in FSWs
	void GetAllNumbersBySTDstage();
	void GetAllPropnsBySTDstage();
	void UpdateSyndromePropns();
	void GetAllSTDcofactors();
	void CalcTransmissionProb();
	void CalcInfectionProb();
	void CalcSTDtransitions();
	void HIVstageChanges();
	void SetHIVnumbersToTemp();
	void GetNewPartners();
	void GetPartnerTransitions();
	void GetAllNumbersRemaining();
	void GetNumbersChanging(SexuallyExpF* a, SexuallyExpF* b, double RatesDep[16][6]);
	void GetAllNumbersChanging();
	void CalcAllAgeChanges();
	void GetNewVirgins();
	void GetTotalBySTDstage();
	void GetTotalGUD();
};

class Child
{
public:
	Child(int Sex);

	int SexInd; // 0 = male, 1 = female
	double PropnBirths;

	PaedHIV Perinatal;
	PaedHIV Breastmilk;
	double NegBirths;
	double HIVneg[10];
	double OnHAART[15];
	double TotalAlive[15]; // Note that from age 10 onwards, only HIV+ kids are included
						   // in this total.
	double NonAIDSmort[15];
	double MortProb1st6m; // Prob of non-AIDS mortality in 1st 6 months of life
	double AIDSdeathsTot[15]; // Total AIDS deaths, by age at END of year

	void Reset();
	void UpdateMort();
	void UpdateBirths();
	void CalcAgeChanges();
};

// --------------------------------------------------------------------------------------
// General functions
// --------------------------------------------------------------------------------------

// Functions for reading input parameters

void ReadSexAssumps(const char* input);
void ReadSTDepi(const char* input);
void ReadRatesByYear();
void ReadMortTables();
void ReadFertTables();
void ReadAgeExitRates();
void ReadOneStartProfileM(ifstream* file, SexuallyExpM* m);
void ReadOneStartProfileF(ifstream* file, SexuallyExpF* f);
void ReadStartProfile(const char* input);
void ReadStartPop();
void ReadSTDprev();
void ReadSexData(const char* input);
void ReadAllInputFiles();
void ReadRandomAdj();

// Other reset functions

double GetDebutRate(int ia, int sex, double start, double tolerance);
double GetPartnerAcqHigh(int sex, double start, double tolerance);
void GetAllPartnerRates();
void GetStartProfile();
void ResetAll();

// Functions called on an annual basis

void OneYearProj();
void UpdateFertAndBirths();
void CalcPrevForLogL();
void UpdateProbCure();
void UpdateSTDtransitionProbs();
void UpdateCondomUse();
void CalcBehavForSumSq();
void UpdateNonAIDSmort();
void CalcAllAgeChanges();
void GetSummary();

// Functions called every sexual behaviour cycle

void OneBehavCycle();
void UpdateMarriageIncidence();
void BalanceSexualPartners();
void GetPartnerAIDSmort();
void CalcPartnerTransitions();

// Functions called every STD cycle

void OneSTDcycle();
void GetAllNumbersBySTDstage();
void UpdateSyndromePropns();
void GetAllSTDcofactors();
void SetMaxHIVprevToFSW();
void GetNewInfections();
void CalcSTDtransitions();

// Functions for calibration and uncertainty analysis

// (a) Functions for SIR analysis
void SimulateParameters();
void SimulateHIVparameters();
void SimulateSexParameters();
void SimulateNGparameters();
void GenerateSample();
void RunSample();

// (b) Functions for MCMC analysis
void ReadStartingPoints();
void ReadCovariance(const char* input);
void ReadOnePrior(ifstream* file, MCMCparameter* a);
void ReadPriors();
void GetCholesky();
void NextMCstep(int ir, int ic);
void GetAcceptanceRate(int n);
void TuneJumpingDbn(int n);
void UpdateCovar(int n);
void CalcConvergence(double pi);
void RecordCovariance();
void Metropolis();
void MetropolisCont(int n);

// (c) Functions for calibration to HIV and STD prevalence data
void SetCalibParameters();
void CalcTotalLogL();
double ReturnNegLogL(double ParameterSet[10]);
void ReadInitSimplex(const char* input, double ParameterCombinations[11][10], int Dimension);
void SaveFinalSimplex(const char* filout, double ParameterCombinations[11][10], int Dimension);
void MaximizeLikelihood(double FTol, const char* input, const char* filout);
void SaveHIVcalibOutput(const char* filout);
void SaveTrend(double Trend[41], const char* filout);
void RecordAllSTDpropns();

// (d) Functions for calibration to sexual behaviour data
void CalcSumSquares();
double ReturnSumSquares(double ParameterSet[16]);
void ReadInitSex(const char* input, double ParameterCombinations[17][16], int Dimension);
void SaveFinalSex(const char* filout, double ParameterCombinations[17][16], int Dimension);
void MinimizeSumSquares(double FTol, const char* input, const char* filout);
void GetSingleSexCalib(double Behav[3][2], int AgeStart, int AgeEnd);
void GetPartnerOutput2005();
double ReturnMarriedPropn(int Sex, int AgeStart, int AgeEnd);
void GetMarriageOutput1996();
void GetMarriageOutput2001();
void GetMarriageOutput2007();
void SaveSexCalibOutput(const char* filout);

// (e) Functions for simultaneous calibration to HIV and sexual behaviour data
void FitHIVandSexData();

// --------------------------------------------------------------------------------------
// Objects created from classes defined above
// --------------------------------------------------------------------------------------

OutputArray RandomUniformHIV(6);
OutputArray HIVtransmProb(6);
OutputArray RandomUniformSex(8);
OutputArray SexBehavParameters(8);
//OutputArray RandomUniformNG(7);
//OutputArray NGparameters(7);
OutputArray LogL(1);

//MCMCparameter lMtoFtransmCSW(NumberSeries); // 'l' prefix denotes logit- or log-transformed
MCMCparameter lMtoFtransmST(NumberSeries);
MCMCparameter lMtoFtransmLT(NumberSeries);
//MCMCparameter lFtoMtransmCSW(NumberSeries);
MCMCparameter lFtoMtransmST(NumberSeries);
MCMCparameter lFtoMtransmLT(NumberSeries);
MCMCparameter lInitHIVprevHigh(NumberSeries);
//MCMCparameter lHighPropnM(NumberSeries);
//MCMCparameter lHighPropnF(NumberSeries);
MCMCparameter lRelPartnerAcqM(NumberSeries);
MCMCparameter lRelPartnerAcqF(NumberSeries);
MCMCparameter lAssortativeness(NumberSeries);
//MCMCparameter lMeanPartnerDur(NumberSeries);
//MCMCparameter lMeanPartnerGapM(NumberSeries);
//MCMCparameter lMeanPartnerGapF(NumberSeries);
MCMCparameter lRelPartnerAcqMM(NumberSeries);
MCMCparameter lRelPartnerAcqMF(NumberSeries);
MCMCparameter lRelPartnerAcqLM(NumberSeries);
MCMCparameter lRelPartnerAcqLF(NumberSeries);
MCMCparameter lCondomBias(NumberSeries);
MCMCparameter lRelARTinfectiousness(NumberSeries);
MCMCparameter lPosterior(NumberSeries);
MCMCparameter AcceptInd(NumberSeries);

PostOutputArray OutTotalPop(26);
PostOutputArray OutTotalHIV(26);
PostOutputArray OutANCbias(1);
PostOutputArray OutLogLStats(2);
//PostOutputArray OutANCbiasTrend(1);
PostOutputArray OutModelVarANC(1);
PostOutputArray OutModelVarHH(2);
PostOutputArray OutModelVarSex(1);
PostOutputArray OutANCprev15(26);
PostOutputArray OutANCprev20(26);
PostOutputArray OutANCprev25(26);
PostOutputArray OutANCprev30(26);
PostOutputArray OutANCprev35(26);
PostOutputArray OutANCprevTot(26);
PostOutputArray OutHSRCprevM(9);
PostOutputArray OutHSRCprevF(9);
PostOutputArray OutHSRC2002M(8);
PostOutputArray OutHSRC2002F(8);
PostOutputArray OutHSRC2008M(9);
PostOutputArray OutHSRC2008F(9);
/*PostOutputArray OutRHRUprev(4);
PostOutputArray OutSexBias(6);
PostOutputArray OutMarried1996(30);
PostOutputArray OutMarried2001(30);
PostOutputArray OutMarried2007(30);
PostOutputArray OutPartnerCalib(30);
PostOutputArray Out15to49prevM(26);
PostOutputArray Out15to49prevF(26);
PostOutputArray OutCSWprev(26);
PostOutputArray OutVirgin(20);
PostOutputArray OutUnmarried0(20);
PostOutputArray OutUnmarried1(20);
PostOutputArray OutUnmarried2(20);
PostOutputArray OutMarried1(20);
PostOutputArray OutMarried2(20);
PostOutputArray OutPopByAge(20);*/
PostOutputArray OutNewHIV(26);
PostOutputArray OutCoVHIVinc(26);
/*PostOutputArray OutHIVinc15to24F(26);
PostOutputArray OutHIVinc25to49F(26);
PostOutputArray OutHIVinc15to24M(26);
PostOutputArray OutHIVinc25to49M(26);*/
PostOutputArray OutHIVinc15to49(26);
PostOutputArray OutHIVincByAgeF(9);
PostOutputArray OutHIVincByAgeM(9);
/*PostOutputArray OutNewHIVMSM(26);
PostOutputArray OutNewHIVMSF(26);
PostOutputArray OutNewHIVMLM(26);
PostOutputArray OutNewHIVMLF(26);
PostOutputArray OutNewHIVUSM(26);
PostOutputArray OutNewHIVUSF(26);
PostOutputArray OutNewHIVCSM(26);
PostOutputArray OutNewHIVCSF(26);*/
/*PostOutputArray OutNewHIVMM(16);
PostOutputArray OutNewHIVUM(16);
PostOutputArray OutNewHIVMF(16);
PostOutputArray OutNewHIVUF(16);
PostOutputArray OutHIVnegMM(16);
PostOutputArray OutHIVnegUM(16);
PostOutputArray OutHIVnegMF(16);
PostOutputArray OutHIVnegUF(16);*/
//PostOutputArray OutAIDSdeaths(26);

HIVtransition HIVtransitionM(0, 0, 0, 0, 0, 0, 0, 18);
HIVtransition HIVtransitionF(1, 0, 0, 0, 0, 0, 45, 18);
SyphilisTransition TPtransitionM(0, 0, 0, 0, 0, 0, 0, 0);
SyphilisTransition TPtransitionF(1, 0, 0, 0, 0, 0, 0, 0);
HerpesTransition HSVtransitionM(0, 0, 0, 0, 0, 0, 0, 0);
HerpesTransition HSVtransitionF(1, 0, 0, 0, 0, 0, 0, 0);
OtherSTDtransition NGtransitionM(0, 0, 0, 0, 0, 8, 0, 0);
OtherSTDtransition NGtransitionF(1, 11, 8, 0, 7, 8, 0, 0);
OtherSTDtransition CTtransitionM(0, 0, 0, 0, 0, 0, 0, 0);
OtherSTDtransition CTtransitionF(1, 0, 0, 0, 0, 0, 0, 0);
OtherSTDtransition HDtransitionM(0, 0, 0, 11, 0, 0, 0, 0);
OtherSTDtransition HDtransitionF(1, 0, 0, 3, 0, 0, 0, 0);
OtherSTDtransition TVtransitionM(0, 0, 0, 0, 0, 0, 0, 0);
OtherSTDtransition TVtransitionF(1, 0, 0, 0, 0, 0, 0, 0);
BVtransition BVtransitionF(0, 0, 0, 0, 0, 0, 0);
VCtransition VCtransitionF(0, 0, 0, 0, 0, 0, 0);

SexCohortM MaleHigh(1);
SexCohortM MaleLow(2);
SexCohortF FemHigh(1);
SexCohortF FemLow(2);
Child MaleChild(0);
Child FemChild(1);
