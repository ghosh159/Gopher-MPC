import random
from pcf.pcf import PCF  # Meta's Private Computation Framework

# Define the function to securely compute average salary
def mpc_average_salary(party_id, salaries, num_parties):
    """
    Securely computes the average salary using PCF primitives.
    
    Args:
        party_id (int): ID of the current party.
        salaries (list): List of salaries (each party has access to its own salary only).
        num_parties (int): Total number of parties in the computation.
    
    Returns:
        float: Securely computed average salary.
    """
    # Step 1: Share the salary as a secret
    secret_salary = PCF.Secret(salaries[party_id])
    
    # Step 2: Add up all shared salaries securely
    total_salary = PCF.Sum(secret_salary)
    
    # Step 3: Compute the average by dividing the total salary by the number of parties
    avg_salary = total_salary / num_parties
    
    # Return the result to all parties
    return avg_salary

# Main function to simulate the computation
def demonstrate_mpc_salary_pcf():
    salaries = [75000, 82000, 68000]  # Private salaries
    num_parties = len(salaries)
    
    print("Original private salaries:", salaries)
    true_avg = sum(salaries) / num_parties
    print("True average:", true_avg)
    
    # Simulate MPC protocol with PCF
    pcf = PCF(num_parties)
    
    # Each party executes the computation
    results = []
    for party_id in range(num_parties):
        result = pcf.run(lambda: mpc_average_salary(party_id, salaries, num_parties))
        results.append(result)
    
    # All parties receive the same result in an MPC setup
    mpc_average = results[0]
    print("MPC-computed average:", mpc_average)
    
    # Verify the result
    assert abs(mpc_average - true_avg) < 1e-10, "MPC calculation error!"
    print("Verification successful: MPC average matches true average")

# Entry point
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run the demonstration
    demonstrate_mpc_salary_pcf()
