
import math
import operator
from statistics import mean, median, mode
import json
import random
import hashlib
import base64
from datetime import datetime
from decimal import Decimal, getcontext
import cmath
from fractions import Fraction

# Set high precision for calculations
getcontext().prec = 50

# Calculator history storage
calculation_history = []

def save_to_history(operation, inputs, result):
    """Save calculation to history"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation,
        "inputs": inputs,
        "result": result
    }
    calculation_history.append(entry)

def show_history():
    """Display calculation history"""
    print("\n🔍 === Calculation History ===")
    if not calculation_history:
        print("No calculations in history.")
        return
    
    for i, entry in enumerate(calculation_history[-15:], 1):  # Show last 15
        print(f"{i}. [{entry['timestamp']}] {entry['operation']}")
        print(f"   📊 Inputs: {entry['inputs']} → 🎯 Result: {entry['result']}")

def export_history():
    """Export calculation history to JSON"""
    filename = f"calculator_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(calculation_history, f, indent=2)
    print(f"💾 History exported to {filename}")

def clear_history():
    """Clear calculation history"""
    global calculation_history
    calculation_history = []
    print("🗑️ History cleared!")

def get_multiple_numbers(min_count=1, max_count=20):
    """Get multiple numbers from user input (flexible count)"""
    while True:
        try:
            count = int(input(f"📊 How many numbers? ({min_count}-{max_count}): "))
            if min_count <= count <= max_count:
                break
            else:
                print(f"❌ Please enter a number between {min_count} and {max_count}!")
        except ValueError:
            print("❌ Please enter a valid number!")
    
    numbers = []
    print(f"🔢 Enter {count} numbers:")
    for i in range(count):
        while True:
            try:
                num = float(input(f"Number {i+1}: "))
                numbers.append(num)
                break
            except ValueError:
                print("❌ Please enter a valid number!")
    return numbers

def quick_sum():
    """Quick sum function that can be used anywhere"""
    print("\n⚡ === Quick Sum ===")
    numbers = get_multiple_numbers(1, 20)
    result = sum(numbers)
    print(f"🔢 Sum of {len(numbers)} numbers: {result}")
    save_to_history("Quick Sum", numbers, result)
    return result

def machine_learning_operations():
    """Basic ML operations without external libraries"""
    print("\n🤖 === Machine Learning Operations ===")
    print("1. 📈 Linear Regression (2D)")
    print("2. 📊 K-Means Clustering (2D)")
    print("3. 🎯 Simple Prediction Model")
    print("4. 📉 Data Correlation Analysis")
    print("5. 🔄 Data Normalization")
    
    choice = input("Choose ML operation (1-5): ")
    
    if choice == '1':  # Linear Regression
        print("📈 Linear Regression - Enter X,Y data pairs:")
        n = int(input("How many data points? "))
        x_vals, y_vals = [], []
        
        for i in range(n):
            x = float(input(f"X{i+1}: "))
            y = float(input(f"Y{i+1}: "))
            x_vals.append(x)
            y_vals.append(y)
        
        # Calculate slope and intercept
        n = len(x_vals)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x*y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x*x for x in x_vals)
        
        slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x*sum_x)
        intercept = (sum_y - slope*sum_x) / n
        
        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean)**2 for y in y_vals)
        ss_res = sum((y - (slope*x + intercept))**2 for x, y in zip(x_vals, y_vals))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1
        
        result = f"📊 y = {slope:.4f}x + {intercept:.4f}, R² = {r_squared:.4f}"
        print(result)
        save_to_history("Linear Regression", [x_vals, y_vals], result)
    
    elif choice == '2':  # K-Means Clustering
        print("📊 K-Means Clustering (2D):")
        n = int(input("How many points? "))
        points = []
        
        for i in range(n):
            x = float(input(f"Point {i+1} X: "))
            y = float(input(f"Point {i+1} Y: "))
            points.append([x, y])
        
        k = int(input("Number of clusters (k): "))
        
        # Simple k-means implementation
        centroids = [[random.uniform(min(p[0] for p in points), max(p[0] for p in points)),
                     random.uniform(min(p[1] for p in points), max(p[1] for p in points))] for _ in range(k)]
        
        for iteration in range(10):  # 10 iterations
            clusters = [[] for _ in range(k)]
            
            # Assign points to nearest centroid
            for point in points:
                distances = [math.sqrt((point[0]-c[0])**2 + (point[1]-c[1])**2) for c in centroids]
                cluster_idx = distances.index(min(distances))
                clusters[cluster_idx].append(point)
            
            # Update centroids
            for i in range(k):
                if clusters[i]:
                    centroids[i] = [sum(p[0] for p in clusters[i])/len(clusters[i]),
                                   sum(p[1] for p in clusters[i])/len(clusters[i])]
        
        print("🎯 Final Centroids:")
        for i, centroid in enumerate(centroids):
            print(f"  Cluster {i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        
        save_to_history("K-Means Clustering", points, centroids)

def cryptography_operations():
    """Cryptographic operations"""
    print("\n🔐 === Cryptography Operations ===")
    print("1. 🔗 Hash Generation (MD5, SHA256)")
    print("2. 🔑 Simple Caesar Cipher")
    print("3. 📊 Base64 Encoding/Decoding")
    print("4. 🎲 Random Number Generation")
    print("5. 🔒 Simple XOR Encryption")
    
    choice = input("Choose crypto operation (1-5): ")
    
    if choice == '1':  # Hash Generation
        text = input("Enter text to hash: ")
        md5_hash = hashlib.md5(text.encode()).hexdigest()
        sha256_hash = hashlib.sha256(text.encode()).hexdigest()
        
        result = f"MD5: {md5_hash}\nSHA256: {sha256_hash}"
        print("🔗 Hash Results:")
        print(result)
        save_to_history("Hash Generation", text, result)
    
    elif choice == '2':  # Caesar Cipher
        text = input("Enter text: ")
        shift = int(input("Enter shift value: "))
        
        encrypted = ""
        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                encrypted += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
            else:
                encrypted += char
        
        print(f"🔑 Encrypted: {encrypted}")
        save_to_history("Caesar Cipher", [text, shift], encrypted)
    
    elif choice == '3':  # Base64
        operation = input("Encode or Decode? (e/d): ").lower()
        text = input("Enter text: ")
        
        if operation == 'e':
            encoded = base64.b64encode(text.encode()).decode()
            print(f"📊 Encoded: {encoded}")
            save_to_history("Base64 Encode", text, encoded)
        else:
            try:
                decoded = base64.b64decode(text).decode()
                print(f"📊 Decoded: {decoded}")
                save_to_history("Base64 Decode", text, decoded)
            except:
                print("❌ Invalid Base64 string!")

def advanced_statistics():
    """Advanced statistical operations"""
    print("\n📊 === Advanced Statistics ===")
    print("1. 📈 Descriptive Statistics")
    print("2. 🎯 Probability Distributions")
    print("3. 📉 Regression Analysis")
    print("4. 🔄 Hypothesis Testing")
    print("5. 📊 Data Visualization (ASCII)")
    
    choice = input("Choose statistical operation (1-5): ")
    
    if choice == '1':  # Descriptive Statistics
        numbers = get_multiple_numbers(3, 20)
        
        # Calculate advanced statistics
        n = len(numbers)
        mean_val = mean(numbers)
        median_val = median(numbers)
        
        # Quartiles
        sorted_nums = sorted(numbers)
        q1 = sorted_nums[n//4] if n >= 4 else sorted_nums[0]
        q3 = sorted_nums[3*n//4] if n >= 4 else sorted_nums[-1]
        iqr = q3 - q1
        
        # Variance and standard deviation
        variance = sum((x - mean_val)**2 for x in numbers) / n
        std_dev = math.sqrt(variance)
        
        # Skewness
        skewness = sum((x - mean_val)**3 for x in numbers) / (n * std_dev**3) if std_dev > 0 else 0
        
        result = f"""
📊 DESCRIPTIVE STATISTICS:
   Count: {n}
   Mean: {mean_val:.4f}
   Median: {median_val:.4f}
   Q1: {q1:.4f}, Q3: {q3:.4f}
   IQR: {iqr:.4f}
   Variance: {variance:.4f}
   Std Dev: {std_dev:.4f}
   Skewness: {skewness:.4f}
   Range: {max(numbers) - min(numbers):.4f}
        """
        print(result)
        save_to_history("Descriptive Statistics", numbers, result)
    
    elif choice == '5':  # ASCII Data Visualization
        numbers = get_multiple_numbers(5, 15)
        
        print("\n📊 ASCII Bar Chart:")
        max_val = max(numbers)
        scale = 50 / max_val if max_val > 0 else 1
        
        for i, num in enumerate(numbers):
            bar_length = int(num * scale)
            bar = "█" * bar_length
            print(f"Data {i+1:2d}: {num:6.2f} |{bar}")
        
        print(f"\nScale: Each █ represents {max_val/50:.2f} units")
        save_to_history("ASCII Visualization", numbers, "Bar chart displayed")

def complex_number_operations():
    """Advanced complex number operations"""
    print("\n🔮 === Complex Number Operations ===")
    print("1. ➕ Complex Addition/Subtraction")
    print("2. ✖️ Complex Multiplication/Division")
    print("3. 📐 Polar Form Conversion")
    print("4. 🌀 Complex Roots")
    print("5. 🎯 Complex Functions")
    
    choice = input("Choose complex operation (1-5): ")
    
    if choice in ['1', '2']:
        print("Enter first complex number:")
        real1 = float(input("Real part: "))
        imag1 = float(input("Imaginary part: "))
        c1 = complex(real1, imag1)
        
        print("Enter second complex number:")
        real2 = float(input("Real part: "))
        imag2 = float(input("Imaginary part: "))
        c2 = complex(real2, imag2)
        
        if choice == '1':
            add_result = c1 + c2
            sub_result = c1 - c2
            result = f"Addition: {add_result}\nSubtraction: {sub_result}"
            print("🔮 Results:")
            print(result)
            save_to_history("Complex Addition/Subtraction", [c1, c2], result)
        
        else:  # choice == '2'
            mul_result = c1 * c2
            div_result = c1 / c2 if c2 != 0 else "undefined"
            result = f"Multiplication: {mul_result}\nDivision: {div_result}"
            print("🔮 Results:")
            print(result)
            save_to_history("Complex Multiplication/Division", [c1, c2], result)
    
    elif choice == '3':  # Polar form
        real = float(input("Real part: "))
        imag = float(input("Imaginary part: "))
        c = complex(real, imag)
        
        magnitude = abs(c)
        phase = cmath.phase(c)
        phase_degrees = math.degrees(phase)
        
        result = f"Rectangular: {c}\nPolar: {magnitude:.4f} ∠ {phase_degrees:.2f}°"
        print("📐 Polar Form:")
        print(result)
        save_to_history("Polar Conversion", c, result)

def high_precision_calculator():
    """Ultra high-precision arithmetic"""
    print("\n🎯 === High-Precision Calculator ===")
    print("🔬 Working with up to 50 decimal places!")
    print("1. ➕ High-Precision Addition")
    print("2. ✖️ High-Precision Multiplication")
    print("3. 📐 High-Precision Square Root")
    print("4. 🔢 High-Precision Power")
    print("5. 📊 Fraction Operations")
    
    choice = input("Choose precision operation (1-5): ")
    
    if choice in ['1', '2', '4']:
        a = Decimal(input("Enter first number: "))
        b = Decimal(input("Enter second number: "))
        
        if choice == '1':
            result = a + b
            operation = "Addition"
        elif choice == '2':
            result = a * b
            operation = "Multiplication"
        else:  # choice == '4'
            result = a ** b
            operation = "Power"
        
        print(f"🎯 High-Precision {operation}: {result}")
        save_to_history(f"High-Precision {operation}", [str(a), str(b)], str(result))
    
    elif choice == '3':
        a = Decimal(input("Enter number: "))
        result = a.sqrt()
        print(f"🔬 High-Precision Square Root: {result}")
        save_to_history("High-Precision Square Root", str(a), str(result))
    
    elif choice == '5':
        print("Enter first fraction:")
        num1 = int(input("Numerator: "))
        den1 = int(input("Denominator: "))
        f1 = Fraction(num1, den1)
        
        print("Enter second fraction:")
        num2 = int(input("Numerator: "))
        den2 = int(input("Denominator: "))
        f2 = Fraction(num2, den2)
        
        result = f"""
🔢 Fraction Operations:
   Addition: {f1 + f2}
   Subtraction: {f1 - f2}
   Multiplication: {f1 * f2}
   Division: {f1 / f2}
        """
        print(result)
        save_to_history("Fraction Operations", [f1, f2], result)

def ai_powered_calculator():
    """AI-powered calculation suggestions"""
    print("\n🤖 === AI-Powered Calculator ===")
    print("🧠 Smart calculation suggestions and pattern recognition!")
    print("1. 🎯 Smart Number Pattern Detection")
    print("2. 🔮 Calculation Prediction")
    print("3. 🧮 Auto-Optimization Suggestions")
    print("4. 📊 Mathematical Insights")
    
    choice = input("Choose AI operation (1-4): ")
    
    if choice == '1':  # Pattern Detection
        numbers = get_multiple_numbers(5, 15)
        
        # Detect arithmetic progression
        differences = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        is_arithmetic = len(set(differences)) == 1
        
        # Detect geometric progression
        if all(n != 0 for n in numbers[:-1]):
            ratios = [numbers[i+1] / numbers[i] for i in range(len(numbers)-1)]
            is_geometric = len(set([round(r, 6) for r in ratios])) == 1
        else:
            is_geometric = False
        
        # Detect fibonacci-like sequence
        is_fibonacci_like = all(numbers[i] + numbers[i+1] == numbers[i+2] 
                               for i in range(len(numbers)-2)) if len(numbers) >= 3 else False
        
        patterns = []
        if is_arithmetic:
            patterns.append(f"🔢 Arithmetic Progression (d = {differences[0]})")
        if is_geometric:
            patterns.append(f"📊 Geometric Progression (r = {ratios[0]:.4f})")
        if is_fibonacci_like:
            patterns.append("🌀 Fibonacci-like Sequence")
        
        if not patterns:
            patterns.append("🤔 No clear pattern detected")
        
        result = "🎯 Pattern Analysis:\n" + "\n".join(patterns)
        print(result)
        save_to_history("Pattern Detection", numbers, result)
    
    elif choice == '2':  # Calculation Prediction
        if len(calculation_history) >= 3:
            recent_ops = [entry['operation'] for entry in calculation_history[-3:]]
            most_common = max(set(recent_ops), key=recent_ops.count)
            
            suggestion = f"🔮 Based on your history, you might want to perform: {most_common}"
            print(suggestion)
            
            # Suggest optimal number ranges
            recent_inputs = []
            for entry in calculation_history[-5:]:
                if isinstance(entry['inputs'], list):
                    if isinstance(entry['inputs'][0], (int, float)):
                        recent_inputs.extend([x for x in entry['inputs'] if isinstance(x, (int, float))])
            
            if recent_inputs:
                avg_magnitude = sum(abs(x) for x in recent_inputs) / len(recent_inputs)
                range_suggestion = f"💡 Suggested number range: {avg_magnitude/2:.2f} to {avg_magnitude*2:.2f}"
                print(range_suggestion)
        else:
            print("🤖 Need more calculation history for predictions!")

def matrix_operations():
    """Advanced matrix operations"""
    print("\n🔢 === Advanced Matrix Operations ===")
    print("1. ➕ Matrix Addition")
    print("2. ✖️ Matrix Multiplication")
    print("3. 📐 Matrix Determinant (3x3)")
    print("4. 🔄 Matrix Transpose")
    print("5. 🎯 Matrix Inverse")
    print("6. 🌟 Eigenvalues (2x2)")
    
    choice = input("Choose matrix operation (1-6): ")
    
    def get_matrix(size):
        matrix = []
        print(f"Enter {size}x{size} matrix:")
        for i in range(size):
            row = []
            for j in range(size):
                val = float(input(f"Element [{i+1}][{j+1}]: "))
                row.append(val)
            matrix.append(row)
        return matrix
    
    if choice == '3':  # 3x3 Determinant
        matrix = get_matrix(3)
        
        # Calculate 3x3 determinant
        det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
               matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
               matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
        
        print(f"📐 3x3 Determinant: {det}")
        save_to_history("3x3 Matrix Determinant", matrix, det)
    
    elif choice == '6':  # Eigenvalues for 2x2
        matrix = get_matrix(2)
        a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        
        # Calculate eigenvalues for 2x2 matrix
        trace = a + d
        det = a * d - b * c
        discriminant = trace**2 - 4*det
        
        if discriminant >= 0:
            lambda1 = (trace + math.sqrt(discriminant)) / 2
            lambda2 = (trace - math.sqrt(discriminant)) / 2
            result = f"🌟 Eigenvalues: λ₁ = {lambda1:.4f}, λ₂ = {lambda2:.4f}"
        else:
            real_part = trace / 2
            imag_part = math.sqrt(-discriminant) / 2
            result = f"🌟 Complex Eigenvalues: λ₁ = {real_part:.4f} + {imag_part:.4f}i, λ₂ = {real_part:.4f} - {imag_part:.4f}i"
        
        print(result)
        save_to_history("Matrix Eigenvalues", matrix, result)

def equation_solver():
    """Advanced equation solver"""
    print("\n🧮 === Advanced Equation Solver ===")
    print("1. 📐 Quadratic Equation")
    print("2. 📊 Cubic Equation")
    print("3. 🔢 System of Linear Equations (3x3)")
    print("4. 🌀 Transcendental Equations (Newton's Method)")
    
    choice = input("Choose equation type (1-4): ")
    
    if choice == '2':  # Cubic Equation
        print("Solve ax³ + bx² + cx + d = 0")
        a = float(input("Enter a: "))
        b = float(input("Enter b: "))
        c = float(input("Enter c: "))
        d = float(input("Enter d: "))
        
        # Simplified cubic solver (real root approximation)
        def cubic_function(x):
            return a*x**3 + b*x**2 + c*x + d
        
        def cubic_derivative(x):
            return 3*a*x**2 + 2*b*x + c
        
        # Newton's method for one real root
        x = 1.0  # Initial guess
        for _ in range(20):
            fx = cubic_function(x)
            if abs(fx) < 1e-10:
                break
            fpx = cubic_derivative(x)
            if fpx == 0:
                break
            x = x - fx / fpx
        
        result = f"🔢 Approximate real root: x ≈ {x:.6f}"
        print(result)
        save_to_history("Cubic Equation", [a, b, c, d], result)

def basic_operations():
    print("\n➕ Basic Operations:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")
    choice = input("Choose operation (1-4): ")

    a = float(input("Enter first number: "))
    b = float(input("Enter second number: "))

    operations = {
        '1': (operator.add, "Addition"),
        '2': (operator.sub, "Subtraction"),
        '3': (operator.mul, "Multiplication"),
        '4': (operator.truediv, "Division")
    }

    if choice in operations:
        op_func, op_name = operations[choice]
        if choice == '4' and b == 0:
            print("❌ Error: Division by zero!")
            return
        result = op_func(a, b)
        print("🎯 Result:", result)
        save_to_history(op_name, [a, b], result)
    else:
        print("❌ Invalid choice!")

def bulk_operations():
    print("\n📊 Bulk Operations (1-20 numbers):")
    print("1. Sum all numbers")
    print("2. Product of all numbers")
    print("3. Average (mean)")
    print("4. Median")
    print("5. Mode")
    print("6. Maximum")
    print("7. Minimum")
    print("8. Range (max - min)")
    print("9. Standard deviation")
    print("10. Square root of each number")
    
    choice = input("Choose operation (1-10): ")
    numbers = get_multiple_numbers(1, 20)
    
    operations = {
        '1': ("Sum", sum(numbers)),
        '2': ("Product", math.prod(numbers)),
        '3': ("Average", mean(numbers)),
        '4': ("Median", median(numbers)),
        '6': ("Maximum", max(numbers)),
        '7': ("Minimum", min(numbers)),
        '8': ("Range", max(numbers) - min(numbers)),
    }
    
    if choice in operations:
        op_name, result = operations[choice]
        print(f"🎯 {op_name}: {result}")
        save_to_history(f"Bulk {op_name}", numbers, result)
    elif choice == '5':
        try:
            result = mode(numbers)
            print(f"🎯 Mode: {result}")
            save_to_history("Bulk Mode", numbers, result)
        except:
            print("❌ No mode found (all numbers are unique)")
    elif choice == '9':
        mean_val = mean(numbers)
        variance = sum((x - mean_val) ** 2 for x in numbers) / len(numbers)
        std_dev = math.sqrt(variance)
        print(f"🎯 Standard deviation: {std_dev}")
        save_to_history("Bulk Standard Deviation", numbers, std_dev)
    elif choice == '10':
        results = [math.sqrt(abs(num)) for num in numbers]
        print("🎯 Square roots:")
        for i, result in enumerate(results):
            print(f"  √|{numbers[i]}| = {result}")
        save_to_history("Bulk Square Roots", numbers, results)
    else:
        print("❌ Invalid choice!")

def advanced_operations():
    print("\n🔬 Advanced Operations:")
    print("1. Square Root")
    print("2. Power")
    print("3. Factorial")
    print("4. Trigonometric Functions")
    print("5. Logarithms")
    print("6. Hyperbolic Functions")
    print("7. Inverse Trigonometric")
    print("8. Gamma Function")
    
    choice = input("Choose operation (1-8): ")

    if choice == '1':
        x = float(input("Enter number: "))
        result = math.sqrt(abs(x))
        print("🎯 Square root:", result)
        save_to_history("Square Root", x, result)
    elif choice == '4':
        angle = float(input("Enter angle in degrees: "))
        rad = math.radians(angle)
        sin_val = math.sin(rad)
        cos_val = math.cos(rad)
        tan_val = math.tan(rad)
        result = f"sin({angle}°) = {sin_val:.6f}, cos({angle}°) = {cos_val:.6f}, tan({angle}°) = {tan_val:.6f}"
        print("🎯 Trigonometric values:")
        print(result)
        save_to_history("Trigonometric", angle, result)
    elif choice == '8':
        x = float(input("Enter number: "))
        result = math.gamma(x)
        print("🎯 Gamma function:", result)
        save_to_history("Gamma Function", x, result)

def calculator_app():
    print("🚀 Welcome to the ULTRA-ADVANCED AI Calculator!")
    print("🧠 Powered by Machine Learning, Cryptography, and High-Precision Computing!")
    
    while True:
        print("\n" + "="*60)
        print("🌟 === ULTRA-ADVANCED AI CALCULATOR SUITE ===")
        print("="*60)
        print("📊 1. Basic Operations (2 numbers)")
        print("🔬 2. Advanced Operations (1 number)")
        print("📈 3. Bulk Operations (1-20 numbers)")
        print("⚡ 4. Quick Sum (1-20 numbers)")
        print("🔢 5. Advanced Matrix Operations")
        print("🧮 6. Advanced Equation Solver")
        print("🤖 7. Machine Learning Operations")
        print("🔐 8. Cryptography Operations")
        print("📊 9. Advanced Statistics")
        print("🔮 10. Complex Number Operations")
        print("🎯 11. High-Precision Calculator")
        print("🧠 12. AI-Powered Calculator")
        print("📜 13. View History")
        print("💾 14. Export History")
        print("🗑️ 15. Clear History")
        print("🚪 16. Exit")
        print("="*60)
        
        choice = input("🎮 Enter your choice (1-16): ")

        try:
            if choice == '1':
                basic_operations()
            elif choice == '2':
                advanced_operations()
            elif choice == '3':
                bulk_operations()
            elif choice == '4':
                quick_sum()
            elif choice == '5':
                matrix_operations()
            elif choice == '6':
                equation_solver()
            elif choice == '7':
                machine_learning_operations()
            elif choice == '8':
                cryptography_operations()
            elif choice == '9':
                advanced_statistics()
            elif choice == '10':
                complex_number_operations()
            elif choice == '11':
                high_precision_calculator()
            elif choice == '12':
                ai_powered_calculator()
            elif choice == '13':
                show_history()
            elif choice == '14':
                export_history()
            elif choice == '15':
                clear_history()
            elif choice == '16':
                print("🎉 Thank you for using the Ultra-Advanced AI Calculator!")
                print("🚀 Keep calculating and exploring mathematics!")
                break
            else:
                print("❌ Invalid option. Try again!")
        
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("🔧 Please try again with valid inputs.")

# Run the ultimate calculator
if __name__ == "__main__":
    calculator_app()
