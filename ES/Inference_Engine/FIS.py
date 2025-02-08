import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyControlSystem:
    def __init__(self):
        # Define input variables
        self.NPEF = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Non productive energy factor')
        self.NPTF = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Non productive time factor')
        self.AEJ = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Average energy per job')
        self.n_i = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Number of jobs')
        self.s2_i = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Energetic variance of a job')
        self.UTR = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Unproductive Time Ratio')

        # Define output variables for each rule base
        self.P_energy = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Priority non-productive energy')
        self.P_time = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Priority non-productive time')
        self.P_prod = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Priority productive energy')

        # Generate membership functions and rules
        self._generate_membership_functions()
        self._define_rules()
        self._create_control_system()

    def _generate_membership_functions(self):
        # Membership functions for NPEF, NPTF, AEJ, n_i, s2_i, UTR
        self.NPEF['low'] = fuzz.trapmf(self.NPEF.universe, [0, 0, 0.30, 0.4])
        self.NPEF['medium'] = fuzz.trimf(self.NPEF.universe, [0.2, 0.5, 0.8])
        self.NPEF['high'] = fuzz.trapmf(self.NPEF.universe, [0.7, 0.9, 1, 1])

        self.NPTF['low'] = fuzz.trapmf(self.NPTF.universe, [0, 0.05, 0.10, 0.15])
        self.NPTF['medium'] = fuzz.trimf(self.NPTF.universe, [0.2, 0.5, 0.8])
        self.NPTF['high'] = fuzz.trapmf(self.NPTF.universe, [0.7, 0.9, 1, 1])

        self.AEJ['low'] = fuzz.trapmf(self.AEJ.universe, [0, 0, 0.30, 0.4])
        self.AEJ['medium'] = fuzz.trimf(self.AEJ.universe, [0.2, 0.5, 0.8])
        self.AEJ['high'] = fuzz.trapmf(self.AEJ.universe, [0.7, 0.9, 1, 1])

        self.n_i['low'] = fuzz.trapmf(self.n_i.universe, [0, 0, 0.30, 0.4])
        self.n_i['medium'] = fuzz.trimf(self.n_i.universe, [0.2, 0.5, 0.8])
        self.n_i['high'] = fuzz.trapmf(self.n_i.universe, [0.7, 0.9, 1, 1])

        self.s2_i['low'] = fuzz.trapmf(self.s2_i.universe, [0, 0, 0.30, 0.4])
        self.s2_i['medium'] = fuzz.trimf(self.s2_i.universe, [0.2, 0.5, 0.8])
        self.s2_i['high'] = fuzz.trapmf(self.s2_i.universe, [0.7, 0.9, 1, 1])

        self.UTR['low'] = fuzz.trapmf(self.s2_i.universe, [0, 0, 0.30, 0.4])
        self.UTR['medium'] = fuzz.trimf(self.s2_i.universe, [0.2, 0.5, 0.8])
        self.UTR['high'] = fuzz.trapmf(self.s2_i.universe, [0.7, 0.9, 1, 1])

        # Membership functions for outputs
        for var in [self.P_energy, self.P_time, self.P_prod]:
            var['very low'] = fuzz.trapmf(var.universe, [0, 0, 0.1, 0.2])
            var['low'] = fuzz.trimf(var.universe, [0.1, 0.3, 0.5])
            var['medium'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
            var['high'] = fuzz.trimf(var.universe, [0.5, 0.7, 0.9])
            var['very high'] = fuzz.trapmf(var.universe, [0.8, 0.9, 1, 1])

    def _define_rules(self):
        # Rule base 1: Minimization of non-productive energy
        rule1_1 = ctrl.Rule(self.NPEF['high'] & self.NPTF['high'], self.P_energy['very high'])
        rule1_2 = ctrl.Rule(self.NPEF['medium'] & self.NPTF['high'], self.P_energy['high'])
        rule1_3 = ctrl.Rule(self.NPEF['low'] & self.NPTF['high'], self.P_energy['medium'])
        rule1_4 = ctrl.Rule(self.NPEF['high'] & self.NPTF['medium'], self.P_energy['high'])
        rule1_5 = ctrl.Rule(self.NPEF['medium'] & self.NPTF['medium'], self.P_energy['medium'])
        rule1_6 = ctrl.Rule(self.NPEF['low'] & self.NPTF['medium'], self.P_energy['low'])
        rule1_7 = ctrl.Rule(self.NPEF['high'] & self.NPTF['low'], self.P_energy['medium'])
        rule1_8 = ctrl.Rule(self.NPEF['medium'] & self.NPTF['low'], self.P_energy['low'])
        rule1_9 = ctrl.Rule(self.NPEF['low'] & self.NPTF['low'], self.P_energy['very low'])

        # Rule base 2: Minimization of non-productive time
        rule2_1 = ctrl.Rule(self.NPTF['high'] & self.UTR['low'], self.P_time['very high'])
        rule2_2 = ctrl.Rule(self.NPTF['medium'] & self.UTR['low'], self.P_time['high'])
        rule2_3 = ctrl.Rule(self.NPTF['low'] & self.UTR['low'], self.P_time['medium'])
        rule2_4 = ctrl.Rule(self.NPTF['high'] & self.UTR['medium'], self.P_time['high'])
        rule2_5 = ctrl.Rule(self.NPTF['medium'] & self.UTR['medium'], self.P_time['medium'])
        rule2_6 = ctrl.Rule(self.NPTF['low'] & self.UTR['medium'], self.P_time['low'])
        rule2_7 = ctrl.Rule(self.NPTF['high'] & self.UTR['high'], self.P_time['low'])
        rule2_8 = ctrl.Rule(self.NPTF['medium'] & self.UTR['high'], self.P_time['low'])
        rule2_9 = ctrl.Rule(self.NPTF['low'] & self.UTR['high'], self.P_time['very low'])

        # Rule base 3: Minimization of productive energy
        rule3_1 = ctrl.Rule(self.AEJ['high'] & self.n_i['high'] & self.s2_i['high'], self.P_prod['very high'])
        rule3_2 = ctrl.Rule(self.AEJ['medium'] & self.n_i['high'] & self.s2_i['high'], self.P_prod['very high'])
        rule3_3 = ctrl.Rule(self.AEJ['low'] & self.n_i['high'] & self.s2_i['high'], self.P_prod['high'])
        rule3_4 = ctrl.Rule(self.AEJ['high'] & self.n_i['medium'] & self.s2_i['high'], self.P_prod['high'])
        rule3_5 = ctrl.Rule(self.AEJ['medium'] & self.n_i['medium'] & self.s2_i['high'], self.P_prod['medium'])
        rule3_6 = ctrl.Rule(self.AEJ['low'] & self.n_i['medium'] & self.s2_i['high'], self.P_prod['low'])
        rule3_7 = ctrl.Rule(self.AEJ['high'] & self.n_i['low'] & self.s2_i['high'], self.P_prod['medium'])
        rule3_8 = ctrl.Rule(self.AEJ['medium'] & self.n_i['low'] & self.s2_i['high'], self.P_prod['low'])
        rule3_9 = ctrl.Rule(self.AEJ['low'] & self.n_i['low'] & self.s2_i['high'], self.P_prod['low'])
        
        rule3_10 = ctrl.Rule(self.AEJ['high'] & self.n_i['high'] & self.s2_i['medium'], self.P_prod['very high'])
        rule3_11 = ctrl.Rule(self.AEJ['medium'] & self.n_i['high'] & self.s2_i['medium'], self.P_prod['high'])
        rule3_12 = ctrl.Rule(self.AEJ['low'] & self.n_i['high'] & self.s2_i['medium'], self.P_prod['medium'])
        rule3_13 = ctrl.Rule(self.AEJ['high'] & self.n_i['medium'] & self.s2_i['medium'], self.P_prod['high'])
        rule3_14 = ctrl.Rule(self.AEJ['medium'] & self.n_i['medium'] & self.s2_i['medium'], self.P_prod['medium'])
        rule3_15 = ctrl.Rule(self.AEJ['low'] & self.n_i['medium'] & self.s2_i['medium'], self.P_prod['low'])
        rule3_16 = ctrl.Rule(self.AEJ['high'] & self.n_i['low'] & self.s2_i['medium'], self.P_prod['medium'])
        rule3_17 = ctrl.Rule(self.AEJ['medium'] & self.n_i['low'] & self.s2_i['medium'], self.P_prod['low'])
        rule3_18 = ctrl.Rule(self.AEJ['low'] & self.n_i['low'] & self.s2_i['medium'], self.P_prod['very low'])
        
        rule3_19 = ctrl.Rule(self.AEJ['high'] & self.n_i['high'] & self.s2_i['low'], self.P_prod['high'])
        rule3_20 = ctrl.Rule(self.AEJ['medium'] & self.n_i['high'] & self.s2_i['low'], self.P_prod['medium'])
        rule3_21 = ctrl.Rule(self.AEJ['low'] & self.n_i['high'] & self.s2_i['low'], self.P_prod['low'])
        rule3_22 = ctrl.Rule(self.AEJ['high'] & self.n_i['medium'] & self.s2_i['low'], self.P_prod['medium'])
        rule3_23 = ctrl.Rule(self.AEJ['medium'] & self.n_i['medium'] & self.s2_i['low'], self.P_prod['medium'])
        rule3_24 = ctrl.Rule(self.AEJ['low'] & self.n_i['medium'] & self.s2_i['low'], self.P_prod['low'])
        rule3_25 = ctrl.Rule(self.AEJ['high'] & self.n_i['low'] & self.s2_i['low'], self.P_prod['medium'])
        rule3_26 = ctrl.Rule(self.AEJ['medium'] & self.n_i['low'] & self.s2_i['low'], self.P_prod['low'])
        rule3_27 = ctrl.Rule(self.AEJ['low'] & self.n_i['low'] & self.s2_i['low'], self.P_prod['very low'])



        # Combine all rules
        self.rules = [
        rule1_1, rule1_2, rule1_3, rule1_4, rule1_5, rule1_6, rule1_7, rule1_8, rule1_9,
        rule2_1, rule2_2, rule2_3, rule2_4, rule2_5, rule2_6, rule2_7, rule2_8, rule2_9,
        rule3_1, rule3_2, rule3_3, rule3_4, rule3_5, rule3_6, rule3_7, rule3_8, rule3_9,
        rule3_10, rule3_11, rule3_12, rule3_13, rule3_14, rule3_15, rule3_16, rule3_17, rule3_18,
        rule3_19, rule3_20, rule3_21, rule3_22, rule3_23, rule3_24, rule3_25, rule3_26, rule3_27]

    def _create_control_system(self):
        # Create control systems for the first three rule bases
        self.P_energy_ctrl = ctrl.ControlSystem(self.rules[:9])  # Rule base 1
        self.P_time_ctrl = ctrl.ControlSystem(self.rules[9:18])  # Rule base 2
        self.P_prod_ctrl = ctrl.ControlSystem(self.rules[18:])   # Rule base 3

        # Create control system simulations for each system
        self.P_energy_simulation = ctrl.ControlSystemSimulation(self.P_energy_ctrl)
        self.P_time_simulation = ctrl.ControlSystemSimulation(self.P_time_ctrl)
        self.P_prod_simulation = ctrl.ControlSystemSimulation(self.P_prod_ctrl)

    def set_input_P_energy(self, npef_value, nptf_value):
        self.P_energy_simulation.input['Non productive energy factor'] = npef_value
        self.P_energy_simulation.input['Non productive time factor'] = nptf_value
        self.P_energy_simulation.compute()
        return self.P_energy_simulation.output['Priority non-productive energy']

    def set_input_P_time(self, nptf_value, UTR_value):
        self.P_time_simulation.input['Unproductive Time Ratio'] = UTR_value
        self.P_time_simulation.input['Non productive time factor'] = nptf_value
        self.P_time_simulation.compute()
        return self.P_time_simulation.output['Priority non-productive time']

    def set_input_P_prod(self, aej_value, n_i_value, s2_i_value):
        self.P_prod_simulation.input['Average energy per job'] = aej_value
        self.P_prod_simulation.input['Number of jobs'] = n_i_value
        self.P_prod_simulation.input['Energetic variance of a job'] = s2_i_value
        self.P_prod_simulation.compute()
        return self.P_prod_simulation.output['Priority productive energy']

class FuzzyCombinedSystem:
    def __init__(self):
        # Define input variables for the combined system
        self.P_e_np = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Priority non-productive energy')
        self.P_t_np = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Priority non-productive time')
        self.P_e_p = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Priority productive energy')

        # Define the output variable for the combined system
        self.P_combined = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Priority combined energy')

        # Generate membership functions and rules for the combined system
        self._generate_membership_functions()
        self._define_combined_rules()
        self._create_combined_control_system()

    def _generate_membership_functions(self):
        # Membership functions for P_e_np, P_t_np, P_e_p
        for var in [self.P_e_np, self.P_t_np, self.P_e_p]:
            var['very low'] = fuzz.trapmf(var.universe, [0, 0, 0.1, 0.2])
            var['low'] = fuzz.trimf(var.universe, [0.1, 0.3, 0.5])
            var['medium'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
            var['high'] = fuzz.trimf(var.universe, [0.5, 0.7, 0.9])
            var['very high'] = fuzz.trapmf(var.universe, [0.8, 0.9, 1, 1])

        # Membership functions for P_combined
        self.P_combined['very low'] = fuzz.trapmf(self.P_combined.universe, [0, 0, 0.1, 0.2])
        self.P_combined['low'] = fuzz.trimf(self.P_combined.universe, [0.1, 0.3, 0.5])
        self.P_combined['medium'] = fuzz.trimf(self.P_combined.universe, [0.3, 0.5, 0.7])
        self.P_combined['high'] = fuzz.trimf(self.P_combined.universe, [0.5, 0.7, 0.9])
        self.P_combined['very high'] = fuzz.trapmf(self.P_combined.universe, [0.8, 0.9, 1, 1])

    def _define_combined_rules(self):
        # Define the rules for the combined system based on the outputs of the previous systems
        rule1 = ctrl.Rule(self.P_e_np['very high'] & self.P_t_np['very high'] & self.P_e_p['very high'], self.P_combined['very high'])
        rule2 = ctrl.Rule(self.P_e_np['very high'] & self.P_t_np['very high'] & self.P_e_p['high'], self.P_combined['very high'])
        rule3 = ctrl.Rule(self.P_e_np['very high'] & self.P_t_np['high'] & self.P_e_p['high'], self.P_combined['high'])
        rule4 = ctrl.Rule(self.P_e_np['high'] & self.P_t_np['high'] & self.P_e_p['medium'], self.P_combined['high'])
        rule5 = ctrl.Rule(self.P_e_np['medium'] & self.P_t_np['high'] & self.P_e_p['high'], self.P_combined['high'])
        rule6 = ctrl.Rule(self.P_e_np['medium'] & self.P_t_np['medium'] & self.P_e_p['high'], self.P_combined['medium'])
        rule7 = ctrl.Rule(self.P_e_np['high'] & self.P_t_np['medium'] & self.P_e_p['medium'], self.P_combined['medium'])
        rule8 = ctrl.Rule(self.P_e_np['medium'] & self.P_t_np['medium'] & self.P_e_p['medium'], self.P_combined['medium'])
        rule9 = ctrl.Rule(self.P_e_np['low'] & self.P_t_np['medium'] & self.P_e_p['medium'], self.P_combined['low'])
        rule10 = ctrl.Rule(self.P_e_np['low'] & self.P_t_np['low'] & self.P_e_p['low'], self.P_combined['very low'])
        rule11 = ctrl.Rule(self.P_e_np['low'] & self.P_t_np['low'] & self.P_e_p['medium'], self.P_combined['low'])
        rule12 = ctrl.Rule(self.P_e_np['medium'] & self.P_t_np['low'] & self.P_e_p['low'], self.P_combined['low'])
        rule13 = ctrl.Rule(self.P_e_np['high'] & self.P_t_np['low'] & self.P_e_p['low'], self.P_combined['medium'])
        rule14 = ctrl.Rule(self.P_e_np['very high'] & self.P_t_np['medium'] & self.P_e_p['low'], self.P_combined['medium'])

        self.combined_rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14]

    def _create_combined_control_system(self):
        # Create the control system for the combined rule base
        self.P_combined_ctrl = ctrl.ControlSystem(self.combined_rules)
        self.P_combined_simulation = ctrl.ControlSystemSimulation(self.P_combined_ctrl)

    def set_input_P_combined(self, p_e_np, p_t_np, p_e_p):
        # Set inputs for the combined rule base
        self.P_combined_simulation.input['Priority non-productive energy'] = p_e_np
        self.P_combined_simulation.input['Priority non-productive time'] = p_t_np
        self.P_combined_simulation.input['Priority productive energy'] = p_e_p
        self.P_combined_simulation.compute()
        return self.P_combined_simulation.output['Priority combined energy']