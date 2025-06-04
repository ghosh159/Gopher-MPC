import crypten
import torch
import crypten.communicator as comm

def calculate_average_salary():
    # Initialize CrypTen
    crypten.init()
    
    # Original salaries (in practice, these would be private to each party)
    salaries = torch.tensor([
        70000.0,  # Party 1's salary
        80000.0,  # Party 2's salary
        90000.0   # Party 3's salary
    ])
    
    # Encrypt the salaries - this creates secret shares
    encrypted_salaries = crypten.cryptensor(salaries)
    
    # Compute encrypted sum and average
    encrypted_sum = encrypted_salaries.sum()
    encrypted_avg = encrypted_sum / 3
    
    # Decrypt only the final result
    average_salary = encrypted_avg.get_plain_text()
    
    return average_salary.item()

if __name__ == "__main__":
    # Run the secure computation
    result = calculate_average_salary()
    print(f"Secure Average Salary: ${result:,.2f}")
    
    # For verification (this would not be done in practice)
    actual_average = (70000 + 80000 + 90000) / 3
    print(f"Actual Average Salary: ${actual_average:,.2f}")