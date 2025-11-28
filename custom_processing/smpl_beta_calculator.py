"""
SMPL Beta Calculator - Python Implementation

This module computes SMPL body shape parameters (betas) from anthropometric measurements
using pre-trained linear regressors. This is a Python port of the Unity C# implementation.

Based on the Unity C# implementation by Pujades et al. 2019
The Virtual Caliper: Rapid Creation of Metrically Accurate Avatars from 3D Measurements
Implemented in Python by Fabian Plum (Imperial College London, EvoBioMech Group & Juelich Research Centre, IAS-7)

Usage:
    calculator = SMPLBetaCalculator()
    betas = calculator.calculate_betas([1.75, 70.0], gender='male')
"""

import numpy as np
from enum import Enum
from typing import List, Union, Optional


class Gender(Enum):
    """Gender enumeration for SMPL model selection."""
    FEMALE = 0
    MALE = 1


class SMPLBetaCalculator:
    """
    Computes SMPL shape beta values from body measurements using linear regression.
    
    Supports 4 different regressors based on the number of input measurements:
    - 2 params: height, weight
    - 4 params: height, weight, armspan, inseam
    - 5 params: height, weight, armspan, inseam, inseam_width
    - 6 params: height, weight, armspan, inseam, inseam_width, wrist_to_shoulder
    
    All measurements should be in meters (for lengths) and kilograms (for weight).
    """
    
    NUM_BETAS = 10
    
    def __init__(self):
        """Initialize the calculator with all pre-trained regressor coefficients."""
        self._initialize_coefficients()
    
    def _initialize_coefficients(self):
        """Initialize all regressor matrices and constants."""
        
        # ============================================================
        # FEMALE COEFFICIENTS
        # ============================================================
        
        # Volume-weight conversion constants
        self._female_a = -2.35648430867
        self._female_b = 1001.43505432
        
        # Regressor 2: [height, vRoot] -> betas
        self._female_A = np.array([
            [12.5455942127, 5.0553042489],
            [10.1758392683, -37.1027483933],
            [1.24469952215, 4.75375379164],
            [-0.542029599023, 3.37980447348],
            [1.45700314783, -4.13427384673],
            [-1.36463259494, 6.33289691775],
            [0.925913696254, -3.13680061231],
            [0.457392221203, 0.216023467283],
            [0.0559419015304, 0.0270701274621],
            [-0.0557684951813, -0.396172003207],
        ])
        
        self._female_B = np.array([
            [-22.729500163],
            [-1.56447178772],
            [-3.99625051267],
            [-0.491353884744],
            [-0.706445124292],
            [-0.345945415946],
            [-0.240311855035],
            [-0.841667628854],
            [-0.103206961499],
            [0.254059051018],
        ])
        
        # Regressor 4: [armspan, height, inseam, vRoot] -> betas
        self._female_A4 = np.array([
            [5.63579016713, 7.60890528036, 0.452722432905, 2.4197141873],
            [-4.35780123887, 10.6876401936, 4.80511344637, -33.213203264],
            [-26.0670648506, 33.3754948041, -16.5940549239, 11.7360074219],
            [-20.8935617795, -29.1869472507, 71.5400886278, 39.4489608787],
            [-2.58651976071, 1.74608175362, 2.87493426825, -1.81745028738],
            [-2.00158048136, -2.58800926954, 4.48164889916, 8.93638690243],
            [-2.03179406026, 1.81687669728, 1.22295621831, -1.68875003894],
            [-5.78484854238, -3.85705329802, 14.1670850974, 8.17668953384],
            [-2.41554163399, -1.03303211433, 4.80431049637, 2.95198573595],
            [-5.20121168337, -3.44146349041, 11.9681495406, 6.4849131455],
        ])
        
        self._female_B4 = np.array([
            [-23.2814628892],
            [-0.397156802718],
            [-3.52614327241],
            [12.0724303702],
            [-0.0103075197805],
            [0.516951398438],
            [0.157796584873],
            [1.82668124406],
            [0.85135827304],
            [2.5426457183],
        ])
        
        # Regressor 5: [armspan, height, inseam, inseamWidth, vRoot] -> betas
        self._female_A5 = np.array([
            [5.58315242701, 7.57770596161, 0.591752635088, -0.395458388155, 2.8180693875],
            [-2.76902362522, 11.6293367613, 0.608731300312, 11.9362159697, -45.2368544108],
            [-25.9007989979, 33.4740435131, -17.033207043, 1.24912707058, 10.4777302449],
            [-21.5947041037, -29.6025266964, 73.3919910626, -5.26756302161, 44.7551098565],
            [-5.8718796399, -0.201209103868, 11.5524391764, -24.6823499522, 23.0457028309],
            [1.60413500383, -0.45083798322, -5.04199994627, 27.0891271296, -18.351173519],
            [-10.032797905, -2.92545967086, 22.3557248299, -60.1101809621, 58.8617500603],
            [-3.79006860009, -2.67471197555, 8.89834335328, 14.9864424062, -6.91953157014],
            [-1.48611004729, -0.48214158848, 2.34943570273, 6.98266142014, -4.08182508021],
            [-3.7995513317, -2.61067463059, 8.26599360381, 10.5304358081, -4.12266021788],
        ])
        
        self._female_B5 = np.array([
            [-23.2576329316],
            [-1.11642216453],
            [-3.60141451722],
            [12.389848858],
            [1.47702811606],
            [-1.11541440958],
            [3.77998063116],
            [0.923612051359],
            [0.430589538164],
            [1.90809137203],
        ])
        
        # Regressor 6: [armspan, height, inseamWidth, inseam, wristToShoulder, vRoot] -> betas
        self._female_A6 = np.array([
            [14.4918254166, 6.61300190847, 1.20421205709, 7.45243288031, -31.3425042765, -4.18787735526],
            [-14.9668401678, 12.9502162089, 9.7459361556, -8.78495944203, 42.914373173, -35.6442637774],
            [-28.7752479493, 33.7853123808, 0.732981620027, -19.2468560522, 10.1128898388, 12.7382505595],
            [-36.8376937116, -27.9518909762, -8.00464406627, 61.6531745404, 53.6279047992, 56.7424814345],
            [-50.919948287, 4.67696461073, -32.7713284245, -23.139640667, 158.488170558, 58.4723449554],
            [-69.7713082422, 7.27827941442, 14.2727192077, -60.0091195212, 251.113172275, 37.7798074596],
            [2.21761388414, -4.25203456247, -57.9104569736, 31.7899198279, -43.0994138339, 49.2277975433],
            [-87.0557115662, 6.34197360153, 0.0349913525243, -55.2255674101, 292.945287565, 58.5621244301],
            [96.1277455855, -11.0525691124, 24.5105244458, 77.5230840995, -343.425187028, -80.8471833686],
            [189.06366582, -23.495482896, 45.1615855666, 156.792363973, -678.531607955, -155.793894484],
        ])
        
        self._female_B6 = np.array([
            [-24.0744495039],
            [0.0019687237899],
            [-3.33786263439],
            [13.7874449465],
            [5.60738614657],
            [5.42884251522],
            [2.65676740072],
            [8.55805515668],
            [-8.51940953195],
            [-15.7751115194],
        ])
        
        # ============================================================
        # MALE COEFFICIENTS
        # ============================================================
        
        # Volume-weight conversion constants
        self._male_a = -5.28069181198
        self._male_b = 1056.44071546
        
        # Regressor 2: [height, vRoot] -> betas
        self._male_A = np.array([
            [-11.5400924739, -6.0210667838],
            [11.4194637569, -45.3828342674],
            [1.33640697715, 3.67473124984],
            [0.00459803156834, -1.23588204264],
            [-1.81353876182, 5.68304846595],
            [-1.99482266027, 10.0631255165],
            [0.570609492393, -2.82022832303],
            [0.18967538558, 0.929038237599],
            [-0.273559389343, 0.0622436919832],
            [0.93626464489, -4.06420392784],
        ])
        
        self._male_B = np.array([
            [23.2009238013],
            [-0.576973330703],
            [-3.98478070278],
            [0.530568503305],
            [0.756108703384],
            [-0.830098157171],
            [0.212038910783],
            [-0.743195634399],
            [0.460625196094],
            [0.102366707097],
        ])
        
        # Regressor 4: [armspan, height, inseam, vRoot] -> betas
        self._male_A4 = np.array([
            [-5.31800336209, -7.29693684696, 0.011406720731, -2.92597394432],
            [-4.45172837311, 8.31978771786, 9.51223184463, -34.4713369665],
            [-24.9197778087, 33.0885553309, -16.9029024293, 3.33107591018],
            [15.8644149399, 22.6844797261, -50.5184675747, -54.6733479749],
            [-1.2524152942, -6.47159517372, 8.08488540688, 13.4887562698],
            [-12.2062633482, -18.4848578216, 37.4978321452, 49.9774985649],
            [3.29686016729, 5.80396751821, -11.2415727263, -14.575974303],
            [-3.75430694444, -3.90648104224, 10.1393555515, 11.9850556095],
            [1.90171527577, -0.141388152957, -2.36062141437, -3.10795574204],
            [-5.17429447819, -1.25727902176, 9.04289703015, 6.8555207938],
        ])
        
        self._male_B4 = np.array([
            [24.0480529161],
            [0.613658470929],
            [-0.874364015716],
            [-4.55456489319],
            [1.36513213964],
            [3.01296084015],
            [-0.88237888489],
            [0.368192515012],
            [0.0382875861558],
            [1.38423991302],
        ])
        
        # Regressor 5: [armspan, height, inseam, inseamWidth, vRoot] -> betas
        self._male_A5 = np.array([
            [-5.1984090408, -7.11780118166, -0.516086205831, 1.98602830178, -4.50463008388],
            [-3.35252437671, 9.96624247484, 4.66398882864, 18.2537951862, -48.9809317266],
            [-24.7009186792, 33.4163758774, -17.8682211958, 3.63445705898, 0.44211509075],
            [18.7482576003, 27.0040750715, -63.2381902859, 47.8901763856, -92.7403387161],
            [-0.485303285807, -5.32256804181, 4.70139567307, 12.7389506698, 3.36280656994],
            [-9.62972558114, -14.6255624797, 26.1335346396, 42.7869556906, 15.9669609339],
            [-0.892707037613, -0.471421651411, 7.23729016697, -69.5735295065, 40.7267015993],
            [-1.78983365367, -0.963973186434, 1.47468189383, 32.6227826826, -13.9461741344],
            [4.75350477523, 4.13019601402, -14.9389677169, 47.3578895345, -40.7518417849],
            [-7.65646164671, -4.97522032803, 19.9909555149, -41.2198020208, 39.6203577667],
        ])
        
        self._male_B5 = np.array([
            [23.8953831199],
            [-0.789545697272],
            [-1.15375168509],
            [-8.23597439925],
            [0.385864623157],
            [-0.27615432506],
            [4.46587147095],
            [-2.13958321993],
            [-3.60220401102],
            [4.55288497658],
        ])
        
        # Regressor 6: [armspan, height, inseamWidth, inseam, wristToShoulder, vRoot] -> betas
        self._male_A6 = np.array([
            [-6.11197136111, -7.10983720445, 2.28878763516, -1.06975095489, 3.09285984449, -4.08191180573],
            [-10.7915639858, 10.0310922838, 20.7191318392, 0.155556391356, 25.1848246986, -45.5387823027],
            [-29.7651304247, 33.4605231305, 5.31276333112, -20.9373884139, 17.1448589803, 2.78539788778],
            [30.3342029028, 26.9030746227, 44.0505334862, -56.2165242694, -39.224149453, -98.1013203811],
            [-10.1578871538, -5.23824731648, 15.9444955777, -1.16067700759, 32.746475607, 7.83844868758],
            [-29.0623577253, -14.4561585583, 49.227031669, 14.3563814445, 65.7890614518, 24.9587161948],
            [-75.1645605536, 0.176043043207, -44.9594486061, -37.7751914389, 251.446921799, 75.0933445034],
            [-11.879702557, -0.876014779865, 35.9666180348, -4.64028652174, 34.1591916317, -9.27744829631],
            [-31.2442974897, 4.44400676395, 59.2877495722, -36.755447791, 121.870347144, -24.095146722],
            [54.948602037, -5.52097981188, -61.9674477215, 57.93277525, -211.949073668, 10.6521044412],
        ])
        
        self._male_B6 = np.array([
            [24.0520893565],
            [0.486496314768],
            [-0.285071413889],
            [-10.2233482717],
            [2.04503355209],
            [3.05718657962],
            [17.2059576083],
            [-0.408836067981],
            [2.57261296066],
            [-6.18595977018],
        ])
    
    def _compute_vroot(self, weight: float, a: float, b: float) -> float:
        """
        Transform weight to volume root representation.
        
        Args:
            weight: Body weight in kg
            a: Gender-specific offset constant
            b: Gender-specific scale constant
            
        Returns:
            Cube root of normalized volume
        """
        v = (weight - a) / b
        return np.cbrt(v)
    
    def _apply_regressor(self, x: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Apply linear regression: betas = A @ x + B
        
        Args:
            x: Input feature vector (n x 1)
            A: Weight matrix (10 x n)
            B: Bias vector (10 x 1)
            
        Returns:
            Beta values (10,)
        """
        return (A @ x + B).flatten()
    
    def _beta_from_height_weight(
        self, 
        height: float, 
        weight: float, 
        a: float, 
        b: float, 
        A: np.ndarray, 
        B: np.ndarray
    ) -> np.ndarray:
        """
        Compute betas from height and weight (2-parameter regressor).
        
        Args:
            height: Body height in meters
            weight: Body weight in kg
            a: Gender-specific volume offset
            b: Gender-specific volume scale
            A: Weight matrix (10 x 2)
            B: Bias vector (10 x 1)
            
        Returns:
            Array of 10 beta values
        """
        vroot = self._compute_vroot(weight, a, b)
        x = np.array([[height], [vroot]])
        return self._apply_regressor(x, A, B)
    
    def _beta_from_height_weight_armspan_inseam(
        self,
        height: float,
        weight: float,
        armspan: float,
        inseam: float,
        a: float,
        b: float,
        A: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Compute betas from height, weight, armspan, and inseam (4-parameter regressor).
        
        Args:
            height: Body height in meters
            weight: Body weight in kg
            armspan: Fingertip-to-fingertip distance in meters
            inseam: Crotch to floor distance in meters
            a: Gender-specific volume offset
            b: Gender-specific volume scale
            A: Weight matrix (10 x 4)
            B: Bias vector (10 x 1)
            
        Returns:
            Array of 10 beta values
        """
        vroot = self._compute_vroot(weight, a, b)
        x = np.array([[armspan], [height], [inseam], [vroot]])
        return self._apply_regressor(x, A, B)
    
    def _beta_from_regressor5(
        self,
        height: float,
        weight: float,
        armspan: float,
        inseam: float,
        inseam_width: float,
        a: float,
        b: float,
        A: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Compute betas from 5 measurements (5-parameter regressor).
        
        Args:
            height: Body height in meters
            weight: Body weight in kg
            armspan: Fingertip-to-fingertip distance in meters
            inseam: Crotch to floor distance in meters
            inseam_width: Hip/pelvis width at crotch level in meters
            a: Gender-specific volume offset
            b: Gender-specific volume scale
            A: Weight matrix (10 x 5)
            B: Bias vector (10 x 1)
            
        Returns:
            Array of 10 beta values
        """
        vroot = self._compute_vroot(weight, a, b)
        x = np.array([[armspan], [height], [inseam], [inseam_width], [vroot]])
        return self._apply_regressor(x, A, B)
    
    def _beta_from_regressor6(
        self,
        height: float,
        weight: float,
        armspan: float,
        inseam: float,
        inseam_width: float,
        wrist_to_shoulder: float,
        a: float,
        b: float,
        A: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Compute betas from 6 measurements (6-parameter regressor).
        
        Args:
            height: Body height in meters
            weight: Body weight in kg
            armspan: Fingertip-to-fingertip distance in meters
            inseam: Crotch to floor distance in meters
            inseam_width: Hip/pelvis width at crotch level in meters
            wrist_to_shoulder: Wrist to shoulder distance in meters
            a: Gender-specific volume offset
            b: Gender-specific volume scale
            A: Weight matrix (10 x 6)
            B: Bias vector (10 x 1)
            
        Returns:
            Array of 10 beta values
        """
        vroot = self._compute_vroot(weight, a, b)
        # Note: order is armspan, height, inseamWidth, inseam, wristToShoulder, vRoot
        x = np.array([[armspan], [height], [inseam_width], [inseam], [wrist_to_shoulder], [vroot]])
        return self._apply_regressor(x, A, B)
    
    def calculate_betas(
        self, 
        measurements: List[float], 
        gender: Union[Gender, str]
    ) -> np.ndarray:
        """
        Calculate SMPL beta values from body measurements.
        
        The regressor is automatically selected based on the number of measurements:
        - 2 measurements: [height, weight]
        - 4 measurements: [height, weight, armspan, inseam]
        - 5 measurements: [height, weight, armspan, inseam, inseam_width]
        - 6 measurements: [height, weight, armspan, inseam, inseam_width, wrist_to_shoulder]
        
        Args:
            measurements: List of body measurements (see above for format)
            gender: Either Gender enum or string ('male' or 'female')
            
        Returns:
            numpy array of 10 beta values
            
        Raises:
            ValueError: If measurements length is invalid or gender is unknown
        """
        # Parse gender
        if isinstance(gender, str):
            gender_lower = gender.lower()
            if gender_lower == 'female':
                gender = Gender.FEMALE
            elif gender_lower == 'male':
                gender = Gender.MALE
            else:
                raise ValueError(f"Unknown gender: {gender}. Use 'male' or 'female'.")
        
        num_measurements = len(measurements)
        if num_measurements not in [2, 4, 5, 6]:
            raise ValueError(
                f"Invalid number of measurements: {num_measurements}. "
                "Expected 2, 4, 5, or 6 measurements."
            )
        
        height = measurements[0]
        weight = measurements[1]
        
        # Select coefficients based on gender
        if gender == Gender.FEMALE:
            a, b = self._female_a, self._female_b
            A2, B2 = self._female_A, self._female_B
            A4, B4 = self._female_A4, self._female_B4
            A5, B5 = self._female_A5, self._female_B5
            A6, B6 = self._female_A6, self._female_B6
        else:
            a, b = self._male_a, self._male_b
            A2, B2 = self._male_A, self._male_B
            A4, B4 = self._male_A4, self._male_B4
            A5, B5 = self._male_A5, self._male_B5
            A6, B6 = self._male_A6, self._male_B6
        
        # Apply appropriate regressor
        if num_measurements == 2:
            return self._beta_from_height_weight(height, weight, a, b, A2, B2)
        
        elif num_measurements == 4:
            armspan = measurements[2]
            inseam = measurements[3]
            return self._beta_from_height_weight_armspan_inseam(
                height, weight, armspan, inseam, a, b, A4, B4
            )
        
        elif num_measurements == 5:
            armspan = measurements[2]
            inseam = measurements[3]
            inseam_width = measurements[4]
            return self._beta_from_regressor5(
                height, weight, armspan, inseam, inseam_width, a, b, A5, B5
            )
        
        else:  # num_measurements == 6
            armspan = measurements[2]
            inseam = measurements[3]
            inseam_width = measurements[4]
            wrist_to_shoulder = measurements[5]
            return self._beta_from_regressor6(
                height, weight, armspan, inseam, inseam_width, wrist_to_shoulder,
                a, b, A6, B6
            )
    
    def calculate_weight(self, volume: float, gender: Union[Gender, str]) -> float:
        """
        Calculate weight from body volume (inverse operation).
        
        Args:
            volume: Body volume in m³
            gender: Either Gender enum or string ('male' or 'female')
            
        Returns:
            Estimated weight in kg
        """
        if isinstance(gender, str):
            gender_lower = gender.lower()
            if gender_lower == 'female':
                gender = Gender.FEMALE
            elif gender_lower == 'male':
                gender = Gender.MALE
            else:
                raise ValueError(f"Unknown gender: {gender}")
        
        if gender == Gender.FEMALE:
            return volume * self._female_b + self._female_a
        else:
            return volume * self._male_b + self._male_a
    
    def get_volume_constants(self, gender: Union[Gender, str]) -> tuple:
        """
        Get the volume-weight conversion constants for a gender.
        
        Args:
            gender: Either Gender enum or string ('male' or 'female')
            
        Returns:
            Tuple of (a, b) where weight = volume * b + a
        """
        if isinstance(gender, str):
            gender_lower = gender.lower()
            if gender_lower == 'female':
                return (self._female_a, self._female_b)
            elif gender_lower == 'male':
                return (self._male_a, self._male_b)
            else:
                raise ValueError(f"Unknown gender: {gender}")
        
        if gender == Gender.FEMALE:
            return (self._female_a, self._female_b)
        else:
            return (self._male_a, self._male_b)


def demo():
    """Demonstrate the SMPLBetaCalculator with example calculations."""
    
    print("=" * 70)
    print("SMPL Beta Calculator - Python Implementation Demo")
    print("=" * 70)
    
    calculator = SMPLBetaCalculator()
    
    # Example 1: 2-parameter regressor (height + weight)
    print("\n--- Example 1: 2-Parameter Regressor (Height + Weight) ---")
    measurements_2 = [1.75, 75.0]  # 1.75m tall, 75kg
    
    print(f"Input: height={measurements_2[0]}m, weight={measurements_2[1]}kg")
    
    betas_male = calculator.calculate_betas(measurements_2, 'male')
    betas_female = calculator.calculate_betas(measurements_2, 'female')
    
    print("\nMale betas:")
    for i, beta in enumerate(betas_male):
        print(f"  β{i}: {beta:12.6f}")
    
    print("\nFemale betas:")
    for i, beta in enumerate(betas_female):
        print(f"  β{i}: {beta:12.6f}")
    
    # Example 2: 4-parameter regressor
    print("\n--- Example 2: 4-Parameter Regressor ---")
    measurements_4 = [1.75, 75.0, 1.78, 0.82]  # +armspan, +inseam
    
    print(f"Input: height={measurements_4[0]}m, weight={measurements_4[1]}kg, "
          f"armspan={measurements_4[2]}m, inseam={measurements_4[3]}m")
    
    betas_4 = calculator.calculate_betas(measurements_4, 'male')
    print("\nMale betas:")
    for i, beta in enumerate(betas_4):
        print(f"  β{i}: {beta:12.6f}")
    
    # Example 3: 6-parameter regressor
    print("\n--- Example 3: 6-Parameter Regressor ---")
    measurements_6 = [1.734, 99.067, 1.893, 0.802, 0.437, 0.4938]
    
    print(f"Input: height={measurements_6[0]}m, weight={measurements_6[1]}kg, "
          f"armspan={measurements_6[2]}m, inseam={measurements_6[3]}m, "
          f"inseam_width={measurements_6[4]}m, wrist_to_shoulder={measurements_6[5]}m")
    
    betas_6 = calculator.calculate_betas(measurements_6, 'male')
    print("\nMale betas:")
    for i, beta in enumerate(betas_6):
        print(f"  β{i}: {beta:12.6f}")
    
    # Example 4: Volume to weight conversion
    print("\n--- Example 4: Volume to Weight Conversion ---")
    volume = 0.075  # m³
    weight_male = calculator.calculate_weight(volume, 'male')
    weight_female = calculator.calculate_weight(volume, 'female')
    
    print(f"Volume: {volume} m³")
    print(f"Estimated male weight: {weight_male:.2f} kg")
    print(f"Estimated female weight: {weight_female:.2f} kg")
    
    # Show volume constants
    print("\n--- Volume-Weight Constants ---")
    a_m, b_m = calculator.get_volume_constants('male')
    a_f, b_f = calculator.get_volume_constants('female')
    print(f"Male:   a = {a_m}, b = {b_m}")
    print(f"Female: a = {a_f}, b = {b_f}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()

