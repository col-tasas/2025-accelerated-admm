import numpy as np
import control as ctrl
from scipy.linalg import block_diag

def build_appended_system(plant, unique_filter):
    """
    Constructs an appended system in ctrl.ss form using the plant and unique filter.
    
    Parameters:
    ----------
    plant : ctrl.ss
        The system representing the plant in state-space form.
    unique_filter : ctrl.ss
        The unique filter constructed from `combine_filters`.
    
    Returns:
    -------
    appended_system : ctrl.ss
        The new appended system in state-space form.
    """
    # Extract matrices from the plant and the unique filter
    A_plant, B_plant, C_plant, D_plant = ctrl.ssdata(plant)
    A_filter, B_filter, C_filter, D_filter = ctrl.ssdata(unique_filter)
    
    # Define the blocks of A_new
    # Top-left: A_plant
    top_left = A_plant

    # Top-right
    top_right = np.zeros((A_plant.shape[0], A_filter.shape[1]))

    # Bottom-left
    half_columns = B_filter.shape[1] // 2
    B_Psi1 = B_filter[:, :half_columns]  
    bottom_left = B_Psi1 @ C_plant 

    # Bottom-right
    bottom_right = A_filter

    # Combine blocks
    A_new = np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])

    # Define B_new
    # Top block
    top_block = B_plant

    # Bottom block
    B_Psi2 = B_filter[:, half_columns:] 
    bottom_block = B_Psi2 + (B_Psi1 @ D_plant)

    # Combine blocks for B_new
    B_new = np.vstack([
        top_block,
        bottom_block
    ])

    # Define C_new
    # First block
    D_Psi1 = D_filter[:, :half_columns] 
    first_block = D_Psi1 @ C_plant 

    # Second block
    second_block = C_filter

    # Combine first and second blocks for C_new
    C_new = np.hstack([
        first_block,
        second_block
    ])

    # Define D_new
    # First term
    D_Psi2 = D_filter[:, half_columns:] 

    # Second term
    second_term = D_Psi1 @ D_plant

    # Combine to form D_new
    D_new = D_Psi2 + second_term

 
    appended_system = ctrl.ss(A_new, B_new, C_new, D_new)
    return appended_system




def combine_filters_different_nonlinearities(num_filters, *filters):
    """
    Important note
    ----------
    the only assumption that is been made here, is that
    the filters for each non-linearities exhibit only one input and output. So, for example,
    considering     u = non-linear(y) ,
    the overal D matrix of filters associated to one non-linearity has two columns,
    the first associated to the input of the non-linearity (y) and the secons column is associated
    to the output (u)
    
    This function
    ------------
    Combines multiple filters in `ctrl.ss` format into a unique filter.

    Parameters:
    ----------
    num_filters : int
        Number of filters to combine.
    *filters : list of ctrl.ss objects
        Filters provided as state-space objects to be combined.
        
    Returns:
    -------
    unique_filter : ctrl.ss
        The unique filter in state-space form combining all the filters.
    """
    if len(filters) != num_filters:
        raise ValueError("The number of filters provided does not match 'num_filters'")
    
    A_list = []  
    C_list = []  
    B1_blocks = []  
    B2_blocks = []  
    D1_blocks = []  
    D2_blocks = []  

    # Extract matrices from each filter
    for i, filter_obj in enumerate(filters, start=1):
        try:
            A, B, C, D = ctrl.ssdata(filter_obj)  # Extract A, B, C, D matrices

            
            A_list.append(A)
            C_list.append(C)

           
            if B.shape[1] >= 1: 
                B1_blocks.append(B[:, 0:1])  
            else:
                B1_blocks.append(np.zeros((B.shape[0], 1))) 

            if B.shape[1] >= 2:  # Ensure B has at least two columns, one for "y" and one for "u"
                B2_blocks.append(B[:, 1:2])
            else:
                B2_blocks.append(np.zeros((B.shape[0], 1)))  

            
            if D.shape[1] >= 1:  # Ensure D has at least one column
                D1_blocks.append(D[:, 0:1])  
            else:
                D1_blocks.append(np.zeros((D.shape[0], 1)))

            if D.shape[1] >= 2:  # Ensure D has at least two columns
                D2_blocks.append(D[:, 1:2]) 
            else:
                D2_blocks.append(np.zeros((D.shape[0], 1)))
        except Exception as e:
            raise ValueError(f"Error extracting matrices from filter {i}: {e}")

    # Create block diagonal A and C matrices
    A_combined, _, _, _ = ctrl.ssdata(ctrl.append(*[ctrl.ss(A, np.zeros((A.shape[0], 1)), np.zeros((1, A.shape[1])), 0) for A in A_list]))
    C_combined = np.block([
        [C if i == j else np.zeros((C.shape[0], A_list[j].shape[0]))
         for j in range(len(A_list))]
        for i, C in enumerate(C_list)])

    # Create custom block matrices for B1 and B2
    B1_combined = np.block([
        [B1 if i == j else np.zeros((B1.shape[0], B1.shape[1])) for j in range(len(B1_blocks))]
        for i, B1 in enumerate(B1_blocks)
    ])
    B2_combined = np.block([
        [B2 if i == j else np.zeros((B2.shape[0], B2.shape[1])) for j in range(len(B2_blocks))]
        for i, B2 in enumerate(B2_blocks)
    ])
    B_combined = np.hstack([B1_combined, B2_combined]) 

    # Create custom block matrices for D1 and D2
    D1_combined = np.block([
        [D1 if i == j else np.zeros((D1.shape[0], D1.shape[1])) for j in range(len(D1_blocks))]
        for i, D1 in enumerate(D1_blocks)
    ])
    D2_combined = np.block([
        [D2 if i == j else np.zeros((D2.shape[0], D2.shape[1])) for j in range(len(D2_blocks))]
        for i, D2 in enumerate(D2_blocks)
    ])
    D_combined = np.hstack([D1_combined, D2_combined])  

    # Print final matrices (debugging output)
    # print("Final Combined Matrices:")
    # print("A_combined:\n", A_combined)
    # print("B_combined:\n", B_combined)
    # print("C_combined:\n", C_combined)
    # print("D_combined:\n", D_combined)

    # Create the unique filter
    unique_filter = ctrl.ss(A_combined, B_combined, C_combined, D_combined)

    return unique_filter


def combine_filters_one_nonlinearity(num_filters, *filters):
    """
    Constructs a new appended system in ctrl.ss form using the input filters.
    
    Parameters:
    ----------
    num_filters : int
        Number of filters to combine.
    *filters : list of ctrl.ss objects
        Filters provided as state-space objects to be combined.
        
    Returns:
    -------
    appended_system : ctrl.ss
        The resulting appended system in state-space form.
    """
    if len(filters) != num_filters:
        raise ValueError("The number of filters provided does not match 'num_filters'")
    
    # Step 1: Extract matrices from each filter
    A_list, B_list, C_list, D_list = [], [], [], []
    for i, filter_obj in enumerate(filters):
        try:
            A, B, C, D = ctrl.ssdata(filter_obj)  # Extract A, B, C, D matrices
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)
            D_list.append(D)
        except Exception as e:
            raise ValueError(f"Error extracting matrices from filter {i+1}: {e}")

    # Step 2: Combine matrices according to the new appending rules
    
    # Combine all A matrices in a diagonal way
    A_new = block_diag(*A_list)

    # Combine all C matrices in a diagonal way
    C_new = block_diag(*C_list)

    # Combine all B matrices by stacking them vertically
    B_new = np.vstack(B_list)

    # Combine all D matrices by stacking them vertically
    D_new = np.vstack(D_list)

    # Step 3: Return the new appended system
    unique_filter = ctrl.ss(A_new, B_new, C_new, D_new)
    
    return unique_filter
