# SMPL Beta Calculator - Linear Regressors Documentation

This document provides comprehensive documentation of the linear regression models used in `SMPLBetaCalculator.cs`, adapted from [Virtual Caliper - BodyCreator](https://virtualcaliper.is.tue.mpg.de/) to compute SMPL body shape parameters (betas) from anthropometric measurements.

## Overview

This variant of an SMPL (Skinned Multi-Person Linear) Model uses 10 shape **betas** (from the first 10 principal components of body shape variation) to control body shape variations. The linear regressors implemented here predict these 10 beta values from simple body measurements, enabling avatar parametrisation from easily obtainable anthropometric data.

### Core Formula

All regressors use the same linear regression formula:

```
β = A × x + B
```

Where:
- **β** (beta): Output vector of 10 shape parameters
- **A**: Weight matrix (10 × n, where n = number of input features)
- **x**: Input feature vector (measurements)
- **B**: Bias vector (10 × 1)

### Weight Transformation

Weight is not used directly. Instead, it's transformed into a "volume root" representation:

```
v = (weight - a) / b
vRoot = v^(1/3)
```

Where `a` and `b` are gender-specific density constants that relate weight to body volume.

---

## Volume-Weight Constants

These constants convert between weight (kg) and body volume (m³):

| Gender | a (offset) | b (scale) |
|--------|------------|-----------|
| **Female** | -2.35648430867 | 1001.43505432 |
| **Male** | -5.28069181198 | 1056.44071546 |

**Inverse formula** (volume → weight):
```
weight = volume × b + a
```

---

## Regressor 2: Height + Weight

### Input Vector (2 features)
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | height | Body height in meters |
| 1 | vRoot | Cube root of normalized volume |

### Female Coefficients

**Matrix A (10×2):**
| Beta | height | vRoot |
|------|--------|-------|
| β₀ | 12.5455942127 | 5.0553042489 |
| β₁ | 10.1758392683 | -37.1027483933 |
| β₂ | 1.24469952215 | 4.75375379164 |
| β₃ | -0.542029599023 | 3.37980447348 |
| β₄ | 1.45700314783 | -4.13427384673 |
| β₅ | -1.36463259494 | 6.33289691775 |
| β₆ | 0.925913696254 | -3.13680061231 |
| β₇ | 0.457392221203 | 0.216023467283 |
| β₈ | 0.0559419015304 | 0.0270701274621 |
| β₉ | -0.0557684951813 | -0.396172003207 |

**Vector B (bias):**
| Beta | Bias |
|------|------|
| β₀ | -22.729500163 |
| β₁ | -1.56447178772 |
| β₂ | -3.99625051267 |
| β₃ | -0.491353884744 |
| β₄ | -0.706445124292 |
| β₅ | -0.345945415946 |
| β₆ | -0.240311855035 |
| β₇ | -0.841667628854 |
| β₈ | -0.103206961499 |
| β₉ | 0.254059051018 |

### Male Coefficients

**Matrix A (10×2):**
| Beta | height | vRoot |
|------|--------|-------|
| β₀ | -11.5400924739 | -6.0210667838 |
| β₁ | 11.4194637569 | -45.3828342674 |
| β₂ | 1.33640697715 | 3.67473124984 |
| β₃ | 0.00459803156834 | -1.23588204264 |
| β₄ | -1.81353876182 | 5.68304846595 |
| β₅ | -1.99482266027 | 10.0631255165 |
| β₆ | 0.570609492393 | -2.82022832303 |
| β₇ | 0.18967538558 | 0.929038237599 |
| β₈ | -0.273559389343 | 0.0622436919832 |
| β₉ | 0.93626464489 | -4.06420392784 |

**Vector B (bias):**
| Beta | Bias |
|------|------|
| β₀ | 23.2009238013 |
| β₁ | -0.576973330703 |
| β₂ | -3.98478070278 |
| β₃ | 0.530568503305 |
| β₄ | 0.756108703384 |
| β₅ | -0.830098157171 |
| β₆ | 0.212038910783 |
| β₇ | -0.743195634399 |
| β₈ | 0.460625196094 |
| β₉ | 0.102366707097 |

---

## Regressor 4: Height + Weight + Armspan + Inseam

### Input Vector (4 features)
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | armspan | Fingertip-to-fingertip distance (meters) |
| 1 | height | Body height in meters |
| 2 | inseam | Crotch to floor distance (meters) |
| 3 | vRoot | Cube root of normalized volume |

### Female Coefficients

**Matrix A (10×4):**
| Beta | armspan | height | inseam | vRoot |
|------|---------|--------|--------|-------|
| β₀ | 5.63579016713 | 7.60890528036 | 0.452722432905 | 2.4197141873 |
| β₁ | -4.35780123887 | 10.6876401936 | 4.80511344637 | -33.213203264 |
| β₂ | -26.0670648506 | 33.3754948041 | -16.5940549239 | 11.7360074219 |
| β₃ | -20.8935617795 | -29.1869472507 | 71.5400886278 | 39.4489608787 |
| β₄ | -2.58651976071 | 1.74608175362 | 2.87493426825 | -1.81745028738 |
| β₅ | -2.00158048136 | -2.58800926954 | 4.48164889916 | 8.93638690243 |
| β₆ | -2.03179406026 | 1.81687669728 | 1.22295621831 | -1.68875003894 |
| β₇ | -5.78484854238 | -3.85705329802 | 14.1670850974 | 8.17668953384 |
| β₈ | -2.41554163399 | -1.03303211433 | 4.80431049637 | 2.95198573595 |
| β₉ | -5.20121168337 | -3.44146349041 | 11.9681495406 | 6.4849131455 |

**Vector B (bias):**
| Beta | Bias |
|------|------|
| β₀ | -23.2814628892 |
| β₁ | -0.397156802718 |
| β₂ | -3.52614327241 |
| β₃ | 12.0724303702 |
| β₄ | -0.0103075197805 |
| β₅ | 0.516951398438 |
| β₆ | 0.157796584873 |
| β₇ | 1.82668124406 |
| β₈ | 0.85135827304 |
| β₉ | 2.5426457183 |

### Male Coefficients

**Matrix A (10×4):**
| Beta | armspan | height | inseam | vRoot |
|------|---------|--------|--------|-------|
| β₀ | -5.31800336209 | -7.29693684696 | 0.011406720731 | -2.92597394432 |
| β₁ | -4.45172837311 | 8.31978771786 | 9.51223184463 | -34.4713369665 |
| β₂ | -24.9197778087 | 33.0885553309 | -16.9029024293 | 3.33107591018 |
| β₃ | 15.8644149399 | 22.6844797261 | -50.5184675747 | -54.6733479749 |
| β₄ | -1.2524152942 | -6.47159517372 | 8.08488540688 | 13.4887562698 |
| β₅ | -12.2062633482 | -18.4848578216 | 37.4978321452 | 49.9774985649 |
| β₆ | 3.29686016729 | 5.80396751821 | -11.2415727263 | -14.575974303 |
| β₇ | -3.75430694444 | -3.90648104224 | 10.1393555515 | 11.9850556095 |
| β₈ | 1.90171527577 | -0.141388152957 | -2.36062141437 | -3.10795574204 |
| β₉ | -5.17429447819 | -1.25727902176 | 9.04289703015 | 6.8555207938 |

**Vector B (bias):**
| Beta | Bias |
|------|------|
| β₀ | 24.0480529161 |
| β₁ | 0.613658470929 |
| β₂ | -0.874364015716 |
| β₃ | -4.55456489319 |
| β₄ | 1.36513213964 |
| β₅ | 3.01296084015 |
| β₆ | -0.88237888489 |
| β₇ | 0.368192515012 |
| β₈ | 0.0382875861558 |
| β₉ | 1.38423991302 |

---

## Regressor 5: Height + Weight + Armspan + Inseam + Inseam Width

### Input Vector (5 features)
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | armspan | Fingertip-to-fingertip distance (meters) |
| 1 | height | Body height in meters |
| 2 | inseam | Crotch to floor distance (meters) |
| 3 | inseamWidth | Hip/pelvis width at crotch level (meters) |
| 4 | vRoot | Cube root of normalized volume |

### Female Coefficients

**Matrix A (10×5):**
| Beta | armspan | height | inseam | inseamWidth | vRoot |
|------|---------|--------|--------|-------------|-------|
| β₀ | 5.58315242701 | 7.57770596161 | 0.591752635088 | -0.395458388155 | 2.8180693875 |
| β₁ | -2.76902362522 | 11.6293367613 | 0.608731300312 | 11.9362159697 | -45.2368544108 |
| β₂ | -25.9007989979 | 33.4740435131 | -17.033207043 | 1.24912707058 | 10.4777302449 |
| β₃ | -21.5947041037 | -29.6025266964 | 73.3919910626 | -5.26756302161 | 44.7551098565 |
| β₄ | -5.8718796399 | -0.201209103868 | 11.5524391764 | -24.6823499522 | 23.0457028309 |
| β₅ | 1.60413500383 | -0.45083798322 | -5.04199994627 | 27.0891271296 | -18.351173519 |
| β₆ | -10.032797905 | -2.92545967086 | 22.3557248299 | -60.1101809621 | 58.8617500603 |
| β₇ | -3.79006860009 | -2.67471197555 | 8.89834335328 | 14.9864424062 | -6.91953157014 |
| β₈ | -1.48611004729 | -0.48214158848 | 2.34943570273 | 6.98266142014 | -4.08182508021 |
| β₉ | -3.7995513317 | -2.61067463059 | 8.26599360381 | 10.5304358081 | -4.12266021788 |

**Vector B (bias):**
| Beta | Bias |
|------|------|
| β₀ | -23.2576329316 |
| β₁ | -1.11642216453 |
| β₂ | -3.60141451722 |
| β₃ | 12.389848858 |
| β₄ | 1.47702811606 |
| β₅ | -1.11541440958 |
| β₆ | 3.77998063116 |
| β₇ | 0.923612051359 |
| β₈ | 0.430589538164 |
| β₉ | 1.90809137203 |

### Male Coefficients

**Matrix A (10×5):**
| Beta | armspan | height | inseam | inseamWidth | vRoot |
|------|---------|--------|--------|-------------|-------|
| β₀ | -5.1984090408 | -7.11780118166 | -0.516086205831 | 1.98602830178 | -4.50463008388 |
| β₁ | -3.35252437671 | 9.96624247484 | 4.66398882864 | 18.2537951862 | -48.9809317266 |
| β₂ | -24.7009186792 | 33.4163758774 | -17.8682211958 | 3.63445705898 | 0.44211509075 |
| β₃ | 18.7482576003 | 27.0040750715 | -63.2381902859 | 47.8901763856 | -92.7403387161 |
| β₄ | -0.485303285807 | -5.32256804181 | 4.70139567307 | 12.7389506698 | 3.36280656994 |
| β₅ | -9.62972558114 | -14.6255624797 | 26.1335346396 | 42.7869556906 | 15.9669609339 |
| β₆ | -0.892707037613 | -0.471421651411 | 7.23729016697 | -69.5735295065 | 40.7267015993 |
| β₇ | -1.78983365367 | -0.963973186434 | 1.47468189383 | 32.6227826826 | -13.9461741344 |
| β₈ | 4.75350477523 | 4.13019601402 | -14.9389677169 | 47.3578895345 | -40.7518417849 |
| β₉ | -7.65646164671 | -4.97522032803 | 19.9909555149 | -41.2198020208 | 39.6203577667 |

**Vector B (bias):**
| Beta | Bias |
|------|------|
| β₀ | 23.8953831199 |
| β₁ | -0.789545697272 |
| β₂ | -1.15375168509 |
| β₃ | -8.23597439925 |
| β₄ | 0.385864623157 |
| β₅ | -0.27615432506 |
| β₆ | 4.46587147095 |
| β₇ | -2.13958321993 |
| β₈ | -3.60220401102 |
| β₉ | 4.55288497658 |

---

## Regressor 6: Height + Weight + Armspan + Inseam + Inseam Width + Wrist-to-Shoulder

### Input Vector (6 features)
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | armspan | Fingertip-to-fingertip distance (meters) |
| 1 | height | Body height in meters |
| 2 | inseamWidth | Hip/pelvis width at crotch level (meters) |
| 3 | inseam | Crotch to floor distance (meters) |
| 4 | wristToShoulder | Wrist to shoulder distance (meters) |
| 5 | vRoot | Cube root of normalized volume |

### Female Coefficients

**Matrix A (10×6):**
| Beta | armspan | height | inseamWidth | inseam | wristToShoulder | vRoot |
|------|---------|--------|-------------|--------|-----------------|-------|
| β₀ | 14.4918254166 | 6.61300190847 | 1.20421205709 | 7.45243288031 | -31.3425042765 | -4.18787735526 |
| β₁ | -14.9668401678 | 12.9502162089 | 9.7459361556 | -8.78495944203 | 42.914373173 | -35.6442637774 |
| β₂ | -28.7752479493 | 33.7853123808 | 0.732981620027 | -19.2468560522 | 10.1128898388 | 12.7382505595 |
| β₃ | -36.8376937116 | -27.9518909762 | -8.00464406627 | 61.6531745404 | 53.6279047992 | 56.7424814345 |
| β₄ | -50.919948287 | 4.67696461073 | -32.7713284245 | -23.139640667 | 158.488170558 | 58.4723449554 |
| β₅ | -69.7713082422 | 7.27827941442 | 14.2727192077 | -60.0091195212 | 251.113172275 | 37.7798074596 |
| β₆ | 2.21761388414 | -4.25203456247 | -57.9104569736 | 31.7899198279 | -43.0994138339 | 49.2277975433 |
| β₇ | -87.0557115662 | 6.34197360153 | 0.0349913525243 | -55.2255674101 | 292.945287565 | 58.5621244301 |
| β₈ | 96.1277455855 | -11.0525691124 | 24.5105244458 | 77.5230840995 | -343.425187028 | -80.8471833686 |
| β₉ | 189.06366582 | -23.495482896 | 45.1615855666 | 156.792363973 | -678.531607955 | -155.793894484 |

**Vector B (bias):**
| Beta | Bias |
|------|------|
| β₀ | -24.0744495039 |
| β₁ | 0.0019687237899 |
| β₂ | -3.33786263439 |
| β₃ | 13.7874449465 |
| β₄ | 5.60738614657 |
| β₅ | 5.42884251522 |
| β₆ | 2.65676740072 |
| β₇ | 8.55805515668 |
| β₈ | -8.51940953195 |
| β₉ | -15.7751115194 |

### Male Coefficients

**Matrix A (10×6):**
| Beta | armspan | height | inseamWidth | inseam | wristToShoulder | vRoot |
|------|---------|--------|-------------|--------|-----------------|-------|
| β₀ | -6.11197136111 | -7.10983720445 | 2.28878763516 | -1.06975095489 | 3.09285984449 | -4.08191180573 |
| β₁ | -10.7915639858 | 10.0310922838 | 20.7191318392 | 0.155556391356 | 25.1848246986 | -45.5387823027 |
| β₂ | -29.7651304247 | 33.4605231305 | 5.31276333112 | -20.9373884139 | 17.1448589803 | 2.78539788778 |
| β₃ | 30.3342029028 | 26.9030746227 | 44.0505334862 | -56.2165242694 | -39.224149453 | -98.1013203811 |
| β₄ | -10.1578871538 | -5.23824731648 | 15.9444955777 | -1.16067700759 | 32.746475607 | 7.83844868758 |
| β₅ | -29.0623577253 | -14.4561585583 | 49.227031669 | 14.3563814445 | 65.7890614518 | 24.9587161948 |
| β₆ | -75.1645605536 | 0.176043043207 | -44.9594486061 | -37.7751914389 | 251.446921799 | 75.0933445034 |
| β₇ | -11.879702557 | -0.876014779865 | 35.9666180348 | -4.64028652174 | 34.1591916317 | -9.27744829631 |
| β₈ | -31.2442974897 | 4.44400676395 | 59.2877495722 | -36.755447791 | 121.870347144 | -24.095146722 |
| β₉ | 54.948602037 | -5.52097981188 | -61.9674477215 | 57.93277525 | -211.949073668 | 10.6521044412 |

**Vector B (bias):**
| Beta | Bias |
|------|------|
| β₀ | 24.0520893565 |
| β₁ | 0.486496314768 |
| β₂ | -0.285071413889 |
| β₃ | -10.2233482717 |
| β₄ | 2.04503355209 |
| β₅ | 3.05718657962 |
| β₆ | 17.2059576083 |
| β₇ | -0.408836067981 |
| β₈ | 2.57261296066 |
| β₉ | -6.18595977018 |

---

## Usage (in the original C# implementation)

### Public Interface

```csharp
// Main calculation function
public bool calculateBetas(float[] measurements, SMPL.Gender gender)

// Retrieve computed betas
public float[] getBetas()

// Calculate weight from volume (inverse operation)
public float calculateWeight(float volume, SMPL.Gender gender)
```

### Measurement Array Format

| Length | Measurements Required |
|--------|----------------------|
| 2 | [height, weight] |
| 4 | [height, weight, armspan, inseam] |
| 5 | [height, weight, armspan, inseam, inseamWidth] |
| 6 | [height, weight, armspan, inseam, inseamWidth, wristToShoulder] |

### Example Usage

```csharp
SMPLBetaCalculator calculator = GetComponent<SMPLBetaCalculator>();
calculator.initialize();

// Using 4-parameter regressor
float[] measurements = new float[] { 1.75f, 70.0f, 1.80f, 0.85f };
calculator.calculateBetas(measurements, SMPL.Gender.MALE);

float[] betas = calculator.getBetas(); // Returns 10 beta values
```

---

## Understanding Beta Parameters

The 10 beta values control different aspects of body shape in the SMPL model (not explicitly encoded, human-named primary variation):

| Beta | Primary Effect (approximate) |
|------|------------------------------|
| β₀ | Overall body size/scale |
| β₁ | Body mass/weight distribution |
| β₂ | Height-related proportions |
| β₃ | Torso/leg ratio |
| β₄ | Body width |
| β₅ | Hip/waist ratio |
| β₆ | Shoulder width |
| β₇ | Limb thickness |
| β₈ | Chest/back depth |
| β₉ | Fine body shape details |

> **Note:** These are learned principal components from body scan data, so each beta actually affects multiple body regions simultaneously.

---

## Interpretation of Weights

### Large Positive Weights
Indicate that increasing the input measurement will increase the corresponding beta value.

### Large Negative Weights
Indicate that increasing the input measurement will decrease the corresponding beta value.

### Near-Zero Weights
Indicate the input has minimal influence on that particular beta.

### Example Analysis (Male Regressor 2, β₁)
- height: +11.42 → Taller people have higher β₁
- vRoot: -45.38 → Heavier people (larger volume) have lower β₁

This suggests β₁ captures something like "tallness relative to weight" or body proportions.

---

## Mathematical Details

### Complete Calculation Example (Regressor 2, Male)

Given:
- height = 1.75 m
- weight = 75 kg

**Step 1: Compute vRoot**
```
v = (75 - (-5.28069181198)) / 1056.44071546
v = 80.28069181198 / 1056.44071546
v = 0.07598...

vRoot = 0.07598^(1/3) = 0.4234...
```

**Step 2: Apply Linear Regression**
```
β₀ = -11.54 × 1.75 + (-6.02) × 0.4234 + 23.20 = 0.62...
β₁ = 11.42 × 1.75 + (-45.38) × 0.4234 + (-0.58) = -0.85...
... (continue for all 10 betas)
```

---

## File Dependencies

- **Matrix Library:** `Assets/ThirdParty/Matrix/Matrix.cs` (LightweightMatrixCSharp)
- **SMPL Core:** `Assets/MPI/SMPL/SMPL.cs` (Gender enum definition)

---

## Notes

1. All measurements should be in **meters** and **kilograms**
2. The regressor automatically selects based on array length (2, 4, 5, or 6 elements)
3. Coefficients are pre-trained on body scan datasets and hardcoded
4. Male and female models use completely separate coefficient sets
5. The cube root transformation of volume helps linearize the relationship between weight and body shape

---

## References

- SMPL: A Skinned Multi-Person Linear Model (Loper et al., 2015)
- Max Planck Institute for Intelligent Systems
- The Virtual Caliper: Rapid Creation of Metrically Accurate Avatars from 3D Measurements (Pujades et al., 2019)

