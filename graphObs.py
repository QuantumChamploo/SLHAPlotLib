import pyslha
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy import linalg as LA

path = os.getcwd()

#todo:
# create more robust file aquitision 
# fill out self.sleptons, self.gauginos ... with cluster organization
# fully flesh out the catagories below


# Graph class wrapper around the slha file to be used
# when parsing data

cluster_thresh = 800
shift = .1
#pdg codes for various particles, grouped as they will be for the graph
higgs = [24, 25, 35, 36, 37]
sleptons = [1000011, 1000013, 1000015, 2000011, 2000013, 2000015, 1000012,
1000014, 1000016]
squarks = [1000001, 1000003, 1000005, 2000001, 2000003, 2000005,
1000002, 1000004, 1000006, 2000002, 2000004, 2000006]
gauginos = [1000021, 1000022, 1000023, 1000024, 1000025, 1000035,
1000037, 1000039]

higgs_anno = {24:"MW", 25:r"$h^0$", 35:r"$H^0$", 36:r"$A^0$", 37:r"$H^\pm$"}
slepton_anno = {1000011:r"$\widetilde{e}_{1}$", 1000013:r"$\widetilde{e}_{2}$", 1000015:r"$\widetilde{e}_{3}$", 2000011:r"$\widetilde{e}_{4}$", 2000013:r"$\widetilde{e}_{5}$", 2000015:r"$\widetilde{e}_{6}$", 1000012:r"$\widetilde{v}_{1}$",
1000014:r"$\widetilde{v}_{2}$", 1000016:r"$\widetilde{v}_{3}$"}
squark_anno = {1000001:r"$\widetilde{d}_{1}$", 1000003:r"$\widetilde{d}_{2}$", 1000005:r"$\widetilde{d}_{3}$", 2000001:r"$\widetilde{d}_{4}$", 2000003:r"$\widetilde{d}_{5}$", 2000005:r"$\widetilde{d}_{6}$",
1000002:r"$\widetilde{u}_{1}$", 1000004:r"$\widetilde{u}_{2}$", 1000006:r"$\widetilde{u}_{3}$", 2000002:r"$\widetilde{u}_{4}$", 2000004:r"$\widetilde{u}_{5}$", 2000006:r"$\widetilde{u}_{6}$",1000011:r"$\widetilde{e}_{L}$"}
gaugino_anno = {1000021:r"$\widetilde{g}$", 1000022:r"$\widetilde{X}^0_1$", 1000023:r"$\widetilde{X}^0_2$", 1000024:r"$\widetilde{X}^\pm_1$", 1000025:r"$\widetilde{X}^0_3$", 1000035:r"$\widetilde{X}^0_4$",
1000037:r"$\widetilde{X}^\pm_2$", 1000039:r"$\widetilde{gr}$"}

cat_dict = {"higgs":higgs_anno,"slepton":slepton_anno,"squark":squark_anno,"gaugino":gaugino_anno}


# mask these groups by an excludes list to have the function to get 
# rid of particles by the users control in a simple way
def clusterFunc3(group):
	hld = []
	hs = []
	hss = []
	for i in group:
		hld.append(i.mass)
	diff = np.diff(hld)
	lss = []
	ls = []
	ls.append(group[0])
	hs.append(group[0].mass)
	for i, d in enumerate(diff):
		if d < cluster_thresh:
			ls.append(group[i+1])
			hs.append(group[i+1].mass)
			if(i == len(hld) - 2):
				lss.append(ls)
				hss.append(hs)
		else:
			lss.append(ls)
			hss.append(hs)
			ls = []
			hs = []
			ls.append(group[i+1])
			hs.append(group[i+1].mass)
			if(i == len(hld) - 2):
				lss.append(ls)
				hss.append(hs)


	return lss

	
# helper function to fix points returns labels for annotation
def fitcluster(clusters):
	anno = []
	parts = []
	for cluster in clusters:
		result_string = ""
		size = len(cluster)
		start = float(-1*(size - 1)/2*shift)
		cat = cat_dict[cluster[0].cat]
		i = 0

		for part in cluster:
			if(i == (len(cluster) - 1)):
				parts.append(part)

			result_string += cat[part.pdg]




			result_string += " "
			part.delta = start
			start += shift
			i+=1
		anno.append(result_string)

	return parts, anno 



class Graph:
	def __init__(self, file,excluded=[],keepNegs=False):

		self.file = pyslha.read(path + "/" + file)
		self.masses = self.file.blocks['MASS'].items()
		self.higgs = [Particle(i[0],i[1]) for i in self.masses if i[0] in higgs if i[0] not in excluded]
		self.sleptons = [Particle(i[0],i[1]) for i in self.masses if i[0] in sleptons if i[0] not in excluded]
		self.squarks = [Particle(i[0],i[1]) for i in self.masses if i[0] in squarks if i[0] not in excluded]
		self.gauginos = [Particle(i[0],i[1]) for i in self.masses if i[0] in gauginos if i[0] not in excluded]

		if(keepNegs==False):
			self.tossNegs()

		self.sleptons.sort(key=lambda x: x.mass)
		self.squarks.sort(key=lambda x: x.mass)
		self.gauginos.sort(key=lambda x: x.mass)
		self.higgs.sort(key=lambda x: x.mass)
		self.ticklabels = []

		self.makeMatrix()

	def tossNegs(self):
		print("in tossNegs")
		for i in self.higgs:
			if (i.mass < 0):
				self.higgs.remove(i)
		for i in self.sleptons:
			if (i.mass < 0):
				self.sleptons.remove(i)
		for i in self.squarks:
			if (i.mass < 0):
				self.squarks.remove(i)
		for i in self.gauginos:
			if (i.mass < 0):
				self.gauginos.remove(i)


	def orgCats(self,includes):
		i = 0
		annotations = []
		annotated_particles = []

		if('higgs' in includes):
			i+=1
			higgs_annos = fitcluster(clusterFunc3(self.higgs))
			annotations.append(higgs_annos[1])
			annotated_particles.append(higgs_annos[0])
			self.ticklabels.append('higgs')
			for j in self.higgs:
				j.x = i
		else:
			self.higgs = []
		if('sleptons' in includes):
			i+=1
			slepton_annos = fitcluster(clusterFunc3(self.sleptons))
			annotations.append(slepton_annos[1])
			annotated_particles.append(slepton_annos[0])
			self.ticklabels.append('sleptons')
			for j in self.sleptons:
				j.x = i
		else:
			self.sleptons = []
		if('gauginos' in includes):
			i+=1
			gaugino_annos = fitcluster(clusterFunc3(self.gauginos))
			annotations.append(gaugino_annos[1])
			annotated_particles.append(gaugino_annos[0])
			self.ticklabels.append('gauginos')
			for j in self.gauginos:
				j.x = i
		else:
			self.gauginos = []

		if('squarks' in includes):
			i+=1
			squark_annos = fitcluster(clusterFunc3(self.squarks))
			annotations.append(squark_annos[1])
			annotated_particles.append(squark_annos[0])
			self.ticklabels.append('squarks')
			for j in self.squarks:
				j.x = i
		else:
			self.squarks = []




		self.fixX()
		return annotations, annotated_particles

	def fixX(self):
		for part in self.higgs:
			part.x = part.x + part.delta
		for part in self.sleptons:
			part.x = part.x + part.delta
		for part in self.squarks:
			part.x = part.x + part.delta
		for part in self.gauginos:
			part.x = part.x + part.delta
	def plot(self,includes=['sleptons','higgs','gauginos','squarks']):

		annotations, annotated_particles = self.orgCats(includes)

		x1 = [i.x for i in self.higgs]
		y1 = [i.mass for i in self.higgs]

		x2 = [i.x for i in self.sleptons]
		y2 = [i.mass for i in self.sleptons]

		x3 = [i.x for i in self.squarks]
		y3 = [i.mass for i in self.squarks]

		x4 = [i.x for i in self.gauginos]
		y4 = [i.mass for i in self.gauginos]

		xs = x1 + x2 + x3 + x4
		ys = y1 + y2 + y3 + y4

		colors = [int((i)*100/len(xs)) for i in range(len(xs))]

		plt.scatter(xs,ys,c=colors, cmap='hsv')




		for x,y in zip(annotations,annotated_particles):
			for i in range(len(x)):

				plt.annotate(x[i],
						xy=(y[i].x,y[i].mass+300),
						ha="center",
						fontsize=8)

		    #label = "{:.2f}".format(y)

		    #plt.annotate(label, # this is the text
		                 #(x,y), # this is the point to label
		                 #textcoords="offset points", # how to position the text
		                 #xytext=(5,0), # distance from text to points (x,y)
		                # ha='left') # horizontal alignment can be left, right or center


		plt.xlim(0,len(includes)+1)

		#plt.semilogy()
		plt.grid(alpha=.5,linestyle="--")
		plt.ylabel("Mass - GeV")

		i = 0
		ticks = [i+1 for i in range(len(includes))]
		

		plt.xticks(ticks)
		plt.axes().set_xticklabels(self.ticklabels)

		#plt.xticks([1,2,3,4])
		#plt.axes().set_xticklabels(['higgs', 'sleptons', 'gauginos', 'squarks'])

		#plt.show()

	def plotBar(self):
		x1 = [higgs_anno[i.pdg] for i in self.higgs]
		y1 = [i.mass for i in self.higgs]

		x2 = [slepton_anno[i.pdg] for i in self.sleptons]
		y2 = [i.mass for i in self.sleptons]

		x3 = [squark_anno[i.pdg] for i in self.squarks]
		y3 = [i.mass for i in self.squarks]

		x4 = [gaugino_anno[i.pdg] for i in self.gauginos]
		y4 = [i.mass for i in self.gauginos]

		xs = x1 + x2 + x3 + x4 
		ys = y1 + y2 + y3 + y4

		#xs = np.arange(len(ys))
		clist = [(0, "red"), (0.125, "red"), (0.25, "orange"), (0.5, "green"), 
         (0.7, "fuchsia"), (0.75, "darkorchid"), (1, "blue")]

		clist2 =[(0,"chartreuse"),(.33,"aqua"),(.66,"magenta"),(1,"mediumpurple")]
		rvb = mcolors.LinearSegmentedColormap.from_list("", clist2)

		colors = [int((i)*100/len(xs)) for i in range(len(xs))]
		color_arr = [float(x/len(xs)) for x in range(len(xs))]
		color_arr = np.array(color_arr)
		plt.bar(xs,ys,color=rvb(color_arr))
		plt.xticks(rotation='vertical',fontsize=6)


		return 


	def plotSimple(self,includes=['sleptons','higgs','gauginos','squarks']):
		annotations, annotated_particles = self.orgCats()
		x1 = [i.x for i in self.higgs]
		y1 = [i.mass for i in self.higgs]

		x2 = [i.x for i in self.sleptons]
		y2 = [i.mass for i in self.sleptons]

		x3 = [i.x for i in self.squarks]
		y3 = [i.mass for i in self.squarks]

		x4 = [i.x for i in self.gauginos]
		y4 = [i.mass for i in self.gauginos]

		xs = x1 + x2 + x3 + x4
		ys = y1 + y2 + y3 + y4

		colors = [int((i)*100/len(xs)) for i in range(len(xs))]

		plt.scatter(xs,ys,c=colors, cmap='hsv')




		for x,y in zip(annotations,annotated_particles):
			for i in range(len(x)):

				plt.annotate(x[i],
						xy=(y[i].x,y[i].mass+300),
						ha="center",
						fontsize=8)

		    #label = "{:.2f}".format(y)

		    #plt.annotate(label, # this is the text
		                 #(x,y), # this is the point to label
		                 #textcoords="offset points", # how to position the text
		                 #xytext=(5,0), # distance from text to points (x,y)
		                # ha='left') # horizontal alignment can be left, right or center


		plt.xlim(0,5)

		#plt.semilogy()
		plt.grid(alpha=.5,linestyle="--")
		plt.ylabel("Mass - GeV")

		plt.xticks([1,2,3,4])
		plt.axes().set_xticklabels(['higgs', 'sleptons', 'gauginos', 'squarks'])

		#plt.show()
	def show(self):
		plt.show()
	def makeMatrix(self):
		try:
			self.Nmat = np.matrix([[self.file.blocks['NMIX'][j+1,i+1] for i in range(int(np.sqrt(len(self.file.blocks['NMIX']))))] for j in range(int(np.sqrt(len(self.file.blocks['NMIX']))))])
			w,v = LA.eigh(self.Nmat)
			self.Nmix = [self.Nmat, w,v]	
		except:
			a = 8	
		try:
			self.Umat = np.matrix([[self.file.blocks['UMIX'][j+1,i+1] for i in range(int(np.sqrt(len(self.file.blocks['UMIX']))))] for j in range(int(np.sqrt(len(self.file.blocks['UMIX']))))])
			w,v = LA.eigh(self.Umat)
			self.Umix = [self.Umat,w,v]
		except:
			a = 8	
		try:
			self.Vmat = np.matrix([[self.file.blocks['VMIX'][j+1,i+1] for i in range(int(np.sqrt(len(self.file.blocks['VMIX']))))] for j in range(int(np.sqrt(len(self.file.blocks['VMIX']))))])
			w,v = LA.eigh(self.Vmat)
			self.USQmix = [self.USQmat,w,v]	
		except:
			a = 8	
		try:
			self.USQmat = np.matrix([[self.file.blocks['USQMIX'][j+1,i+1] for i in range(int(np.sqrt(len(self.file.blocks['USQMIX']))))] for j in range(int(np.sqrt(len(self.file.blocks['USQMIX']))))])
			self.USQmix = [self.USQmat,w,v]
			w,v = LA.eigh(self.DSQmat)		
		except:
			a = 8	
		try:
			self.DSQmat = np.matrix([[self.file.blocks['DSQMIX'][j+1,i+1] for i in range(int(np.sqrt(len(self.file.blocks['DSQMIX']))))] for j in range(int(np.sqrt(len(self.file.blocks['DSQMIX']))))])
			w,v = LA.eigh(self.DSQmat)
			self.DSQmix = [self.DSQmat,w,v]		
		except:
			a = 8	
		try:
			self.SELmat = np.matrix([[self.file.blocks['SELMIX'][j+1,i+1] for i in range(int(np.sqrt(len(self.file.blocks['SELMIX']))))] for j in range(int(np.sqrt(len(self.file.blocks['SELMIX']))))])
			w,v = LA.eigh(self.SELmat)
			self.SELmix = [self.SELmat,w,v]
		except:
			a = 8	
		try:
			self.SNUmat = np.matrix([[self.file.blocks['SNUMIX'][j+1,i+1] for i in range(int(np.sqrt(len(self.file.blocks['SNUMIX']))))] for j in range(int(np.sqrt(len(self.file.blocks['SNUMIX']))))])
			w,v = LA.eigh(self.SNUmat)
			self.SNUmix = [self.SNUmat,w,v]
		except:
			a = 8	

class Particle:
	def __init__(self,pdg,mass):
		self.pdg = pdg
		self.mass = mass
		self.delta = 0


		if (int(pdg) in higgs):
			self.cat = "higgs"
			#self.x = 1
		elif (int(pdg) in sleptons):
			self.cat = "slepton"
			#self.x = 2
		elif (int(pdg) in squarks):
			self.cat = "squark"
			#self.x = 4
		elif (int(pdg) in gauginos):
			self.cat = "gaugino"
			#self.x = 3
		else:
			self.cat = "Na"
			print ("created a particle of unknown type")






