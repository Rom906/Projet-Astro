from constants import UNIT, UNIT_V


class ScientificNotation:
    """
    Class for converting numbers to scientific notation with automatic unit management.
    
    This class normalizes a number in scientific notation (mantissa between 1 and 10) and adjusts
    the appropriate unit to keep the exponent between -3 and 3 (e.g., 6371000 m → 6.37 × 10³ km).
    
    Supported units: km, m, mm, um, nm, pm (from largest to smallest)
    """
    
    def __init__(self, number, unit="m"):
        """
        Initialize a ScientificNotation instance.
        
        Args:
            number (float): The number to convert to scientific notation
            unit (str): The unit of the number. Default is "m" (meter). 
                       Must be one of: km, m, mm, um, nm, pm
        """
        self.number = number
        self.unit = unit

    def to_scientific_notation(self, velocity=False):
        """
        Convert the number to scientific notation with unit adjustment.
        
        Process:
        1. If number is 0, return "0"
        2. Normalize the mantissa to be between 1 and 10 (divide by 10 and increment exponent)
        3. Adjust unit upward (km) if exponent >= 3 (reduce exponent by 3)
        4. Normalize mantissa if it is < 1 (multiply by 10 and decrement exponent)
        5. Adjust unit downward (mm) if exponent < -3 (increase exponent by 3)
        
        Returns:
            str: Formatted string in scientific notation (e.g., "6.37 x 10^3 km")
        """
        # Special case: zero
        if not velocity:
            if self.number == 0:
                return "0"
        
            exponent = 0
            mantissa = self.number
            
            # STEP 1: Initial normalization - get mantissa between 1 and 10
            while abs(mantissa) >= 10:
                mantissa /= 10
                exponent += 1
            
            # STEP 2: Adjust units upward (km) if exponent >= 3
            # Changing from m → km reduces exponent by 3
            while exponent >= 3 and self.unit in UNIT and self.unit != "km":
                    self.unit = UNIT[UNIT.index(self.unit) - 1]  # Index -1 = larger unit
                    exponent -= 3
                    
            # STEP 3: Fine normalization - if mantissa < 1, adjust
            while abs(mantissa) < 1:
                mantissa *= 10
                exponent -= 1
            
            # STEP 4: Adjust units downward (mm) if exponent < -3
            # Changing from m → um increases exponent by 3
            while exponent < -3 and self.unit in UNIT and self.unit != "mm":
                    self.unit = UNIT[UNIT.index(self.unit) + 1]  # Index +1 = smaller unit
                    exponent += 3
        else:
            if self.number == 0:
                return "0"
        
            exponent = 0
            mantissa = self.number
            
            # STEP 1: Initial normalization - get mantissa between 1 and 10
            while abs(mantissa) >= 10:
                mantissa /= 10
                exponent += 1
            
            # STEP 2: Adjust units upward (km.s^-1) if exponent >= 3
            # Changing from m.s^-1 → km.s^-1 reduces exponent by 3
            while exponent >= 3 and self.unit in UNIT_V and self.unit != "km.s^-1":
                    self.unit = UNIT_V[UNIT_V.index(self.unit) - 1]  # Index -1 = larger unit
                    exponent -= 3
                    
            # STEP 3: Fine normalization - if mantissa < 1, adjust
            while abs(mantissa) < 1:
                mantissa *= 10
                exponent -= 1
            
            # STEP 4: Adjust units downward (mm.s^-1) if exponent < -3
            # Changing from m.s^-1 → um.s^-1 increases exponent by 3
            while exponent < -3 and self.unit in UNIT_V and self.unit != "mm.s^-1":
                    self.unit = UNIT_V[UNIT_V.index(self.unit) + 1]  # Index +1 = smaller unit
                    exponent += 3
    
        # Return formatted string with 2 decimal places for mantissa
        return f"{mantissa:.2f} x 10^{exponent} {self.unit}"