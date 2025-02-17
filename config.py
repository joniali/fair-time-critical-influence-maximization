import numpy as np

class infMaxConfig(object):

	def __init__(self):

		self.synthetic1    =  True	
		self.facebook      =  False
		self.facebook_rice =  False
		self.instagram     =  False

		if self.synthetic1:

			self.num_nodes = 500

			self.weight = 0.05 # on the edges

			self.p_with = 0.025 # within group edge probability

			self.p_acrosses = [0.001, 0.025, 0.015, 0.005] # list of across group edge probablity

			self.p_across =0.001 # 0.001, 0.015, 0.01, 0.025 # current across group probablity

			self.group_ratios = [0.5,0.6,0.7,0.8]  # experiments for dataset params 

			self.group_ratio = 0.7 # group size ratio 

			# self.gammas_log = [1.0]#[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]

			# self.gamma_log = 1.0

			self.gammas_root = [2.0]#[2.0,6.0,10.0] # root value i.e., 1/2.0

			# self.gammas_root_majority = [1.1, 1.2, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5, 10.0]

			# self.beta_root = [1.0]#,2.0,3.0, 4.0 ] 

			# self.gamma_root = 2.0

			self.seed_size = 30 # sead budget  

			# self.types = [1,2] # type of the algorithm 1 is for standar influence maximizaiton and 2 is for log fairnes formulation 

			# self.type = 2

			self.filename = 'results/synthetic_data'

			self.reach_list = [0.1,0.2,0.3]
			self.reach = 0.2 # budget for the set cover problem

			# self.gamma_timings_a_list = [1.0,0.9,0.8,0.7,0.6]

			# self.gamma_timings_b_list = [0.0]

			self.tau = 20 #time deadline 

			self.num_groups = 2 # number of social groups to generate
			
			self.batch_size = 100

			self.cpu_process_factor = 4 # factor for pralellization 

		

		elif self.facebook:

			self.weight = 0.01
			self.filename = 'facebook/facebook_combined' 
			self.filename_communities = 'facebook/cmtyvv' 
			self.num_groups = 5
			self.seed_size = 30
			self.tau = 20 #self.num_nodes
			self.batch_size = 60
			self.cpu_process_factor = 2
			self.gammas_log = [1.0]#, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]

			self.gamma_log = 1.0

			self.gammas_root = [2.0]#,6.0,10.0]

			self.reach_list = [0.1]#,0.2]#, 0.2, 0.3]
			self.reach = 0.1

			self.beta_root = [1.0]

		elif self.facebook_rice:

			self.weight = 0.01#0.01 # 0.02 tau deadline thing maybe not necessary
			self.filename = 'facebook_rice_lg/rice-univ-facebook-links.elist' 
			self.filename_communities = 'facebook_rice_lg/rice-facebook-undergrads-users.attr' 
			self.num_groups = 4
			self.seed_size = 30 
			self.tau = 20 # , 100, 50, 20]1300 5 
			
			self.gammas_log = [1.0]#, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]

			self.gamma_log = 1.0

			self.gammas_root = [2.0]#,6.0,10.0]

			self.reach_list = [0.1]#, 0.2, 0.3]

			self.reach = 0.1#, 0.2, 0.3]


			self.beta_root = [1.0]

			self.batch_size = 60

			self.cpu_process_factor = 2

		
		elif self.instagram:

			self.weight = 0.06 #0.04 0.06 0.08( reduced) 0.1 tau deadline thing maybe not necessary
			self.filename = 'instagram/instagram_edges.csv' 
			self.filename_communities = 'instagram/instagram_groups.csv' 
			self.num_groups = 2
			self.seed_size = 30
			self.tau = 2 #3 # , 100, 50, 20]1300 5 tau = 3 -> weight 0.04 rest was 0.01
			
			self.gammas_log = [1.0]#, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]

			self.gamma_log = 1.0

			self.gammas_root = [2.0]#,6.0,10.0]

			self.reach_list = [0.0015, 0.002, 0.0025]

			self.reach = 0.0015

			self.beta_root = [1.0]

			self.num_groups = 2

			self.batch_size = 400

			self.cpu_process_factor = 5

