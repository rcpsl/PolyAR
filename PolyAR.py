# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:57:40 2020

@author: Wael
"""

from z3 import *
from yices import *
from scipy.optimize import minimize
from autograd import elementwise_grad as egrad
from autograd import jacobian
from autograd import grad
import itertools 
import numpy as np
import polytope as pc
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue
from functools import partial
import time






class PolyInequalitySolver:

    # ========================================================
    #   Constructor
    # ========================================================
    def __init__(self, num_vars, pregion):
        self.num_vars = num_vars
        self.pregion = pregion
        
        self.poly_inequality_coeffs  = []

        self.__SMT_region_bounds      = 0.1    # threshold below which we use SMT solver/CAD to compute the sign of the polynomial
        
        self.ambiguous_regions_polysss=[]# ambiguous regions of all polys in poly_inequality_coeffs
        self.sign_regions_polysss=[] # sign regions of all polys in poly_inequality_coeffs
        self.type_regions=[] # types of regions in self.sign_regions_polysss ('P' or 'N')
        self.status=False

    

    # ========================================================
    #   Add A Polynomial Inequality
    # ========================================================
    def addPolyInequalityConstraint(self, poly):
        
        self.poly_inequality_coeffs.append(poly)
        
        
        
    

    # ========================================================
    #   Output s string in this form:'x0 x1...xn'
    # ========================================================
    
    def strVari(self,n):
        variables=''
        for i in range(0,n):
            variables=variables+'x'+str(i)+' '
        c= variables[:-1]  
        return c   
    
    # ========================================================
    #   Output s string in this form: '(* val xi)' 
    # ========================================================
    
    def strterm(self,val,i):
        valstr=str(val)
        istr=str(i)
        term='(*'+' '+valstr+' '+'x'+istr+')'
        return term
    
    # ========================================================
    #   Output s string in this form: '(* coef (^ xi i)....' 
    # ========================================================
    
    def strtermpoly(self,coef,pows):
        coefstr=str(coef)
        termpoly='(* '+coefstr
        
        for i in range(len(pows)):
            if pows[i] !=0:
                istr=str(i)
                powstr=str(pows[i])
                termpoly=termpoly+' '+'(^ '+'x'+istr+' '+powstr+')'
            
        termpoly=termpoly+')'    
        return termpoly
        
    # ========================================================
    #   1) Output list of fmla bounds string for yices2
    # ========================================================
    
    def fmlabounds1(self,pregion):
        fmlastrlist=[]
        A=pregion[0]['A']
        b=pregion[0]['b']
        
        for i in range(len(A[:,1])):
            fmlastr='(<(+'
            strb=str(b[i])
            aux=''
            for j in range(self.num_vars):
                aux=aux+self.strterm(A[i,j],j)
                
            fmlastr=fmlastr+aux+')'+''+strb+')'    
            fmlastrlist.append(fmlastr)
            
        return fmlastrlist 
    
    # ========================================================
    #   2) Output list of fmla bounds string for yices2
    # ========================================================
    
    def fmlabounds2(self,box):

        
        fmlastr='(and'
        for i in range(self.num_vars):
            strb1=str(box[0][i][0])
            strb2=str(box[1][i][0])
             
            fmlastr=fmlastr+'( >= '+'x'+str(i)+' '+strb1+')'+''+'( <= '+'x'+str(i)+' '+strb2+')'   

            
        fmlastr=fmlastr+')'  
        return fmlastr 
    
    
    # ========================================================
    #   Output fmla poly string for yices2
    # ========================================================
    
    def fmlapoly(self,poly,sign):
        
        if sign=='N':
            fmlastrpoly='(<=(+'
            for monomial_counter in range(0,len(poly)):
                coeff = poly[monomial_counter]['coeff']
                vars  = poly[monomial_counter]['vars']
    
                
                pows=[]
                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    pows.append(power)
                
                fmlastrpoly=fmlastrpoly+' '+ self.strtermpoly(coeff,pows)
            
            fmlastrpoly=fmlastrpoly+')'+' '+'0)'
            
        else:
            fmlastrpoly='(>=(+'
            for monomial_counter in range(0,len(poly)):
                coeff = poly[monomial_counter]['coeff']
                vars  = poly[monomial_counter]['vars']
    
                
                pows=[]
                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    pows.append(power)
                
                fmlastrpoly=fmlastrpoly+' '+ self.strtermpoly(-coeff,pows)
            
            fmlastrpoly=fmlastrpoly+')'+' '+'0)'
        return fmlastrpoly 
    
        
    # ========================================================
    # Compute the approx Lipchtz constant L of multivar poly 
    # in region
    # ========================================================    
    def Lipchtz(self,poly,region,num_samples):
        all_list=[]
        for i in range(self.num_vars):
            X=np.linspace(region[i]['min'], region[i]['max'], num_samples, endpoint=True)
            all_list.append(X)
        all_coords=list(itertools.product(*all_list))
        all_coordsarray=[]
        for i in range(len(all_coords)):
            all_coordsarray.append(list(all_coords[i]))
        all_coordsarray=np.array(all_coordsarray)   
        
        poly_vals=[]
        for i in range(len(all_coordsarray)):
            poly_vals.append(self.evaluate_multivar_poly(poly,all_coordsarray[i]))
        poly_vals=np.array(poly_vals) 
        poly_vals_diff=np.diff(poly_vals)
        all_coordsarray_diff=np.diff(all_coordsarray,axis=0)
        all_coordsarray_diff=np.linalg.norm(all_coordsarray_diff,axis=1)
        L=max(abs(poly_vals_diff)/all_coordsarray_diff)
        return L
    
    # ========================================================
    # Partition region into subregions around the components 
    # that have higher rate change threshold
    # ========================================================  
    def Partition_ratechange(self,poly,polype,num_samples):
        
        ambiguous_regions=[]
        boundingbox=pc.bounding_box(polype)
        
        all_list=[]
        for i in range(self.num_vars):
            X=np.linspace(boundingbox[0][i][0], boundingbox[1][i][0], num_samples, endpoint=True)
            all_list.append(X)
            
        sampldist=all_list[0][1]- all_list[0][0]   
        
        all_coords=list(itertools.product(*all_list))
        all_coordsarray=[]
        for i in range(len(all_coords)):
            all_coordsarray.append(list(all_coords[i]))
        all_coordsarray=np.array(all_coordsarray)  
        
        ratechange=[]
        for i in range(len(all_coordsarray)):
            ratechange.append(np.linalg.norm(self.Gradient(poly,all_coordsarray[i])))
            
        res = [] 
        for idx in range(0, len(ratechange)) : 
            if ratechange[idx] > 50000000: 
                res.append(idx) 
                
        if len(res)==0:
            return 0, ambiguous_regions

        highratechangev=all_coordsarray[res,:]

        ps=[]
        for i in range(len(highratechangev)):
            box=[]
            for j in range(self.num_vars):
                box.append([highratechangev[i][j]-sampldist,highratechangev[i][j]+sampldist])    
            box=np.array(box)
            p=pc.box2poly(box)
            p=p.intersect(polype)
            ps.append(p)
            
        psreg=pc.Region(ps)  
        pambig=(polype.diff(psreg)).union(psreg) 

        for polytope in pambig:
            ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
        
        return 1,ambiguous_regions
        #return 1,ratechange
        
        
        
        
        
        
    
    # ========================================================
    # Partition polype to 2 polypes along the long dimension 
    # ========================================================    
    def Part_polype(self,polype):
        box=pc.bounding_box(polype)
        boxn=np.append(box[0], box[1], axis=1)
        indexmax=np.argmax(box[1]-box[0])
        mid=0.5*(boxn[indexmax][0]+boxn[indexmax][1])
        
        box1=boxn
        box2=boxn
        box1=np.delete(box1, indexmax, axis=0)
        box1=np.insert(box1, indexmax, np.array([boxn[indexmax][0],mid]), axis=0)
        
        box2=np.delete(box2, indexmax, axis=0)
        box2=np.insert(box2, indexmax, np.array([mid,boxn[indexmax][1]]), axis=0)
        
        p1=pc.box2poly(box1)
        p2=pc.box2poly(box2)
        
        p1=polype.intersect(p1)
        p2=polype.intersect(p2)
        
        return p1,p2

    # ========================================================
    #   Compute the Hessian Matrix of multivar poly at point x
    # ========================================================
    def Hessian(self,poly,x):
        
        def multivar_poly(x):
            result = 0
            for monomial_counter in range(0,len(poly)):
                coeff = poly[monomial_counter]['coeff']
                vars  = poly[monomial_counter]['vars']
                product = coeff
                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    var   = x[var_counter]  
                    product = product * (var**power)
                result = result + product
            return result
    
        H_f = jacobian(egrad(multivar_poly))  
        return H_f(x)
    
    # ========================================================
    #   Compute the Gradient Vector of multivar poly at point x
    # ========================================================
    def Gradient(self,poly,x):
    
        def multivar_poly(x):
            result = 0
            for monomial_counter in range(0,len(poly)):
                coeff = poly[monomial_counter]['coeff']
                vars  = poly[monomial_counter]['vars']
                product = coeff
                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    var   = x[var_counter]  
                    product = product * (var**power)
                result = result + product
            return result    
    
        grad_f = grad(multivar_poly)  
        return grad_f(x)
    
    
    # ========================================================
    #   Evaluate the multivar poly at point x
    # ========================================================
    
    def evaluate_multivar_poly(self,poly, x):
        result = 0
        for monomial_counter in range(0,len(poly)):
            coeff = poly[monomial_counter]['coeff']
            vars  = poly[monomial_counter]['vars']
            product = coeff
            for var_counter in range(len(vars)):
                power = vars[var_counter]['power']
                var   = x[var_counter]  
                product = product * (var**power)
            result = result + product
        return result
    
    
    # ========================================================
    #   Check if a matrix M is positive semidefinite or not
    # ========================================================
    def is_pos_sem_def(self,M):
        return np.all(np.linalg.eigvals(M) >= 0)
    
    # ========================================================
    #   Check if a matrix M is positive definite or not
    # ========================================================
    def is_pos_def(self,M):
        return np.all(np.linalg.eigvals(M) > 1e-8)
    
    # ========================================================
    #   Check if a matrix M is negative definite or not
    # ========================================================
    def is_neg_def(self,M):
        return np.all(np.linalg.eigvals(M) < 0)
    
    # ========================================================
    #   Number of positive eigenvalues: Lambda_i>0
    # ========================================================
    def num_pos_eig(self,M):
        w=np.linalg.eigvals(M)
        return np.sum(w > 0)
    
        
    # ========================================================
    #   Compute the upper bound of 1st ord Remin of Taylor app
    # ========================================================
    def remainder1cst(self,poly,pregion,mid_point,Gradi):
        
        cons=[{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
        
        objectiveFunction = lambda x:-abs(self.evaluate_multivar_poly(poly,x)-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))))
        res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
        regneg=res.fun
                    
        stat=res.status
        if stat==0:
            return -regneg
        else:
            b=[]
            return b
    
    
    # ========================================================
    #   Compute the upper bound of 2nd ord Remin of Taylor app
    # ========================================================
    
    
    def remainder2cst(self,poly,pregion,mid_point,Gradi,Hess):
        
        cons=[{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
        
        objectiveFunction = lambda x:-abs(self.evaluate_multivar_poly(poly,x)-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))))
        res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
        regneg=res.fun
        stat=res.status
        if stat==0:
            return -regneg
        else:
            b=[]
            return b
    
        

    
    
 
   
        
   # ========================================================
    # Output:the rectangle incribed in polytope: A,b
    # ========================================================     
    def RecinPolytope(self,A,b):
        Ap=A.clip(0)
        Am=(-A).clip(0)
        cons=[{'type':'ineq','fun':lambda x:b-(Ap.dot(x[0:self.num_vars]))+(Am.dot(x[self.num_vars:]))},{'type':'ineq','fun':lambda x:x[0:self.num_vars]-x[self.num_vars:]-0.001}]
        
    
         
        objectiveFunction = lambda x: -np.prod(x[0:self.num_vars]-x[self.num_vars:])
        res = minimize(objectiveFunction, np.ones(2*(self.num_vars))/(2*(self.num_vars)), constraints=cons, options={'disp': False})
        regneg=res.x
                    
        stat=res.status
        if stat==0:
            b=(np.array([regneg[self.num_vars:]]).T,np.array([regneg[0:self.num_vars]]).T)
            return b
        else:
            b=[]
            return b     
    
    
 
   
        
    # ========================================================
    # Output:Vertex v of under-approx polytope tangent to tem (one sheet case)
    # ========================================================     
    def Ver_tang_one_sheet(self,poly,pregion,mid_point,Hess,Gradi,Rem2,Tem,sign):

        if sign=='N':    
            cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))+Rem2+0.01)},{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]

            
    
        
            objectiveFunction = lambda x: -x.dot(Tem)
            res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
            regneg=res.x
                        
            stat=res.status
            if stat==0:
                return regneg
            else:
                b=[]
                return b  
            
        else:   
            cons=[{'type':'ineq','fun':lambda x:(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))-Rem2)},{'type':'ineq','fun':lambda x:-pregion[0]['b']+(pregion[0]['A']).dot(x)}]

                
        
            objectiveFunction = lambda x: -x.dot(Tem)
            res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
            regneg=res.x
                        
            stat=res.status
            if stat==0:
                return regneg
            else:
                b=[]
                return b 
    # ========================================================
    # Output:Vertex v of under-approx polytope tangent to tem
    # (two sheets case): The left side of the hyperplane:
    #                   Ax<=b
    # ========================================================     
    def Ver_tang_two_sheet(self,poly,pregion,mid_point,Hess,Gradi,Rem2,Tem,A,b,sign):

        if sign=='N':    
            cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))+Rem2)},{'type':'ineq','fun':lambda x:b-A.dot(x)},{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
    #        cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,x))}]
            
        
            objectiveFunction = lambda x: -x.dot(Tem)
            res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
            # np.random.uniform(low=lowvals, high=highvals, size=self.num_vars)
            regneg=res.x
                        
            stat=res.status
           # print(stat)
            if stat==0:
                return regneg
            else:
                b=[]
                return b   
        else:
            cons=[{'type':'ineq','fun':lambda x:(self.evaluate_multivar_poly(poly,mid_point)+Gradi.dot((x-mid_point))+0.5*(x-mid_point).dot(Hess.dot((x-mid_point)))-Rem2)},{'type':'ineq','fun':lambda x:-b+A.dot(x)},{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
    #        cons=[{'type':'ineq','fun':lambda x:-(self.evaluate_multivar_poly(poly,x))}]
            
        
            objectiveFunction = lambda x: -x.dot(Tem)
            res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), constraints=cons, options={'disp': False})
            # np.random.uniform(low=lowvals, high=highvals, size=self.num_vars)
            regneg=res.x
                        
            stat=res.status
           # print(stat)
            if stat==0:
                return regneg
            else:
                b=[]
                return b
            
            
        
    # ========================================================
    #   The main solver that solve the multivar constraints
    # ========================================================
    def solve(self):
        if self.poly_inequality_coeffs == []:
            print('ERROR: At least one polynomial constraint is needed')
            return self.bounds[0]['min']
        

        poly_list=list(range(0,len(self.poly_inequality_coeffs)))

        
        # for poly_index in range(len(self.poly_inequality_coeffs)):
            
        #     poly_listaux=[i for i in poly_list if i != poly_index]           
        #     negative_regions=self.sign_regions_polysss[poly_index]
            
        #     if not negative_regions:
                
        #         for i in range(len(negative_regions)):
                    
        #             region=negative_regions[i]
        #             negpoint=self.Quad_Prog_mult_var(region,poly_listaux,type_regionsaux)
        #             if len(negpoint)!=0:
        #                 print(negpoint)
        #                 return 'SAT'
            
            
        aux2=self.ambiguous_regions_polysss
        for i in range(len(aux2)):

            aux3=aux2.pop(0)
    
            for ccc in range(len(aux3)):
                regionf=aux3.pop(0)
    
                Areg=regionf[0]['A']
                breg=regionf[0]['b']
                polype=pc.Polytope(Areg, breg) 
                
                boxxpolype=pc.bounding_box(polype)
                boxxpolype=np.append(boxxpolype[0], boxxpolype[1], axis=1)
                polype=pc.box2poly(boxxpolype)
                
    
                
                ###########################################################################
                p1,p2=self.Part_polype(polype)
                plist=[[p1,p2]]
                kk=0
                while kk<1:
                    plistsub=[]
                    for j in range(len(plist[kk])):
    
                        p1,p2=self.Part_polype(plist[kk][j])
                        plistsub=plistsub+[p1,p2]
                    plist.append(plistsub)   
                    kk=kk+1
    
                iterable=[]
                for kkk in range(len(plist[0])):   
                    boxx=pc.bounding_box(plist[0][kkk])   
                    iterable.append(boxx)
                    

    
    
                
                # print('start')    
                procs=[]
                q=Queue()
                for box in iterable:
                        
                    proc= Process(target=self.Yicesmany_multivars, args=(self.poly_inequality_coeffs,box,q))    
                    proc.start()
                    procs.append(proc)
                    
                    
                is_done = True
                counter=0
                while is_done:
                      time.sleep(0.01)
                      for process in procs:
                          if (not process.is_alive()):
                              if (not q.empty()):
                                  is_done = False
                                  self.status=True
                                  break
                              else:
                                  counter=counter+1    
                      if counter==len(procs):
                          break 
                      
                      counter=0
    
                                
                      
                                
                if self.status:                
                    # print(q.get())
                    for process in procs:
                        process.terminate()
                    return 'SAT', q.get()
               
                else:
                    for process in procs:
                        process.terminate()
                
                

        return 'UNSAT'
                    


           

        
    #========================================================
    #   Construct the constraints for the Quadratic Programm
    #========================================================      
        
    def MultVarCons(self,poly,pregion,point,x,sign):
        Hess=self.Hessian(poly,point)
        Gradi=self.Gradient(poly,point)
        
        if sign=='N':
            if self.is_pos_sem_def(Hess):# Hessian is sem def pos: Keep the poly
                remainder2=self.remainder2cst(poly,pregion,point,Gradi,Hess)
                out=self.evaluate_multivar_poly(poly,point)+Gradi.dot((x-point))+0.5*(x-point).dot(Hess.dot((x-point)))+remainder2
                return out
            else: # Hessian is not sem def pos: Keep the poly: Taylor overapprox 1st order
                remainder1=self.remainder1cst(poly,pregion,point,Gradi)
                out=self.evaluate_multivar_poly(poly,point)+Gradi.dot((x-point))+remainder1
                return out 
        else:
            if self.is_pos_sem_def(Hess):# Hessian is sem def pos: Keep the poly
                remainder2=self.remainder2cst(poly,pregion,point,Gradi,Hess)
                out=-self.evaluate_multivar_poly(poly,point)+Gradi.dot((x-point))-0.5*(x-point).dot(Hess.dot((x-point)))-remainder2
                return out
            else: # Hessian is not sem def pos: Keep the poly: Taylor overapprox 1st order
                remainder1=self.remainder1cst(poly,pregion,point,Gradi)
                out=-self.evaluate_multivar_poly(poly,point)-Gradi.dot((x-point))-remainder1
                return out 
    
        
        
    
 
    # ========================================================
    #   Applying Multivar Quad Prog to find feasible point
    # ======================================================== 

    def Quad_Prog_mult_var(self,pregion,poly_list):
        
        # Transform pregion into polytope format
        Areg=pregion[0]['A']
        breg=pregion[0]['b']
        polype=pc.Polytope(Areg, breg) 
        
        # Compute the middle point in the polytope region
        rb,mid_point=pc.cheby_ball(polype)
#        print('mid_point')
#        print(mid_point)
        
        consN=[{'type':'ineq','fun':lambda x, poly=poly:-self.MultVarCons(self.poly_inequality_coeffs[poly],pregion,mid_point,x,'N')} for poly in poly_list if self.type_regions[poly]=='N']
        consP=[{'type':'ineq','fun':lambda x, poly=poly:self.MultVarCons(self.poly_inequality_coeffs[poly],pregion,mid_point,x,'P')} for poly in poly_list if self.type_regions[poly]=='P']
        
        cons=consN+consP+[{'type':'ineq','fun':lambda x:pregion[0]['b']-(pregion[0]['A']).dot(x)}]
        

    


    
        objectiveFunction = lambda x: 0  
        res = minimize(objectiveFunction, np.ones(self.num_vars)/(self.num_vars), method='SLSQP', constraints=cons, options={'disp': False})
        regneg=res.x
        
       
        stat=res.status
        if stat==0:
            return regneg
        else:
            b=[]
            return b      
    
    
    
        
    # ========================================================
    #   Iterative alg to partition Ambig reg until gets small
    # ======================================================== 
    def iterat(self,aux_ambigreg,poly_counter,sign):

        aux_neg_pos_reg=[] 
        aux_ambigreg_res=aux_ambigreg 
        while aux_ambigreg_res:
             region = aux_ambigreg_res.pop(0)
             polype=pc.Polytope(region[0]['A'], region[0]['b']) 
            # print(region)
             if polype.volume < self.__SMT_region_bounds:
                 
                 aux_ambigreg_res.append(region)

                 return  aux_ambigreg_res, aux_neg_pos_reg
             
             else:
                 if sign=='N':
                     ambiguous_regions, negative_regions=self.Npartition_inequality_regions(poly_counter, region)
    
                     aux_ambigreg_res=aux_ambigreg_res+ambiguous_regions 
                     aux_neg_pos_reg=aux_neg_pos_reg+negative_regions

                 else:

                     ambiguous_regions, positive_regions=self.Ppartition_inequality_regions(poly_counter, region)

                     aux_ambigreg_res=aux_ambigreg_res+ambiguous_regions 
                     aux_neg_pos_reg=aux_neg_pos_reg+positive_regions


        return  aux_ambigreg_res, aux_neg_pos_reg         
     
 
            
        
            
        
   # ========================================================
    # Partition the region (Neg+Ambig) based on the polynomial sign
    # ========================================================  
    def Npartition_inequality_regions(self,inequality_index, pregion): 
        num_samples=3
        poly=self.poly_inequality_coeffs[inequality_index]
        
        # Initialization of the neg, ambig, regions
        negative_regions        = []
        ambiguous_regions       = [] 
        # Transform pregion into polytope format
        Areg=pregion[0]['A']
        breg=pregion[0]['b']
        polype=pc.Polytope(Areg, breg) 
        

        
        # Compute the middle point in the polytope region
        rb,mid_point=pc.cheby_ball(polype)
     
        # Compute the hessian Matrix of Poly at mid_point
        Hess=self.Hessian(poly,mid_point+1)
    
        # Compute the gradient vector of Poly at mid_point
        Gradi=self.Gradient(poly,mid_point)
        # Compute the remainder for the 2nd order Taylor overapproximation
        Rem2=self.remainder2cst(poly,pregion,mid_point,Gradi,Hess)
        
        if self.num_pos_eig(Hess)==1: # There is two sheets
            # Compute the hyperplane (As,bs) that will separate the two sheets
              
            #  Compute the center of the hyperbola
            coefcen=Hess.dot(mid_point)-Gradi
            #print(Hess)
            cen=np.linalg.solve(Hess,coefcen)
            # Compute the principal axis As
            eigval,eigvec=np.linalg.eig(Hess)
            # Compute the index of the eigenvalue > 0
            indexaux=np.nonzero(eigval> 0)
            index=indexaux[0][0]
            # Compute As and bs
            As=eigvec[index,:]
            bs=As.dot(cen)
             
            # Compute 2n vertices (2n faces) of the two polytope that under-approximate the negative regions if they exist      
            # Construct 2n template vectors 
            Tem1=np.identity(self.num_vars)
            Tem2=-np.identity(self.num_vars)
            Tem=np.insert(Tem1,self.num_vars,Tem2,axis=0)
        
            # Compute the vertices of the two polytopes  
            vertices1=[]
            vertices2=[]
            for i in range((self.num_vars)+1):

                v1=self.Ver_tang_two_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],As,bs,'N')
                v2=self.Ver_tang_two_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],-As,-bs,'N')
                v1=list(v1)
                v2=list(v2)
            
            
                # if the vertices exist==> Neg Reg exist==> Save it
               # print(v1,v2)
                if (len(v1)!=0) and (len(v2)!=0):
                    vertices1.append(v1)
                    vertices2.append(v2)
                elif (len(v1)!=0):  
                    vertices1.append(v1)
                elif (len(v2)!=0):  
                    vertices2.append(v2)    
                else: # if the vertices  does not exist==> Neg Reg does not exist==> Partition Ambiguous Region
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)

                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, negative_regions
        
                    
            vertices1=np.array(vertices1)
            vertices2=np.array(vertices2)
            
            
            
            
            if (len(vertices1)!=0) and (len(vertices2)!=0): 
                # Compute the two polytopes p1 and p2 using the H-representations
                p1 = pc.qhull(vertices1)
                p2 = pc.qhull(vertices2)          
                # Compute H-representations of  p1 and p2
                A1=p1.A
                b1=p1.b
                A2=p2.A
                b2=p2.b               
                # Compute the polytope p3 the union of p1 and p2 (it presents the negative region)
                if ((p1.volume==0) and (p2.volume==0)):
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, negative_regions
                    
                    
                elif ((p1.volume!=0) and (p2.volume==0)):
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1) 
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    negative_regions.append([{'A':p1.A,'b':p1.b}])
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p1)
                    p4=polype.diff(p1) 
                    
                elif ((p1.volume==0) and (p2.volume!=0)):
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    negative_regions.append([{'A':p2.A,'b':p2.b}])
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p2)
                    p4=polype.diff(p2)
                    
                elif ((p1.volume!=0) and (p2.volume!=0)):
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1)  
                    p1=pc.box2poly(boxunrec1)
                    negative_regions.append([{'A':p1.A,'b':p1.b}])
                    
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1)  
                    p2=pc.box2poly(boxunrec2)
                    negative_regions.append([{'A':p2.A,'b':p2.b}])
                    p3=p2.union(p1)             
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p3)
                    p4=polype.diff(p3) 
                    
                    
 
                # Compute the H-repren of each polytope in p4
                if len(p4)==0:
                    if p4.volume==0:
                        ambiguous_regions=ambiguous_regions+[]
                    else:
                        ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                else:
                    for polytope in p4:
                        if polytope.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                
                return ambiguous_regions, negative_regions
            
            elif (len(vertices1)!=0) and (len(vertices2)==0):           
                # Compute the polytope p1 
                p1= pc.qhull(vertices1) 
                # Compute H-representations of  p1 
                A1=p1.A
                b1=p1.b   
                
                if p1.volume==0:
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, negative_regions
                    
                    
                    
                else:
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    negative_regions.append([{'A':p1.A,'b':p1.b}])                    
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p1)
                    p4=polype.diff(p1)     
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                                
                    return ambiguous_regions, negative_regions

                
            elif (len(vertices1)==0) and (len(vertices2)!=0):     
                # Compute the polytope p2 
                p2= pc.qhull(vertices2) 
                # Compute H-representations of  p2 
                A2=p2.A
                b2=p2.b   
                
                if p2.volume==0:
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, negative_regions
                    
                    
                    
                else:
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    negative_regions.append([{'A':p2.A,'b':p2.b}])                    
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p2)
                    p4=polype.diff(p2)     
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                                
                    return ambiguous_regions, negative_regions
 
 


        else: # There is only one sheet  
            # Compute 2n vertices (2n faces) of the polytope that under-approximate the negative region if it exists         
            # Construct 2n template vectors 
            Tem1=np.identity(self.num_vars)
            Tem2=-np.identity(self.num_vars)
            Tem=np.insert(Tem1,self.num_vars,Tem2,axis=0)
        
            # Compute the vertices   
            vertices=[]
            for i in range((self.num_vars)+1):


                v=self.Ver_tang_one_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],'N')
                v=np.around(v,1)
                v=list(v)
                # if the vertice exist==> Neg Reg exist==> Save it
                if len(v)!=0:
                    print(vertices)
                    if (len(vertices)!=0) and (v in vertices):
                        p1,p2=self.Part_polype(polype)
                        pambig=p1.union(p2)
                        # Compute Aambig and bambig of each region in the partition region
                        if len(pambig)==0:
                            if pambig.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])
    
                            
                        else:
                            for polytope in pambig:
                                if polytope.volume==0:
                                    ambiguous_regions=ambiguous_regions+[]
                                else:
                                    ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                        return ambiguous_regions, negative_regions
                        
                    else:    
                        vertices.append(v)
                else: # if the vertice  does not exist==> Neg Reg does not exist==> Partition Ambiguous Region
                    # create a polytope that is inside pregion formed by a center and n vertices
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    # Compute Aambig and bambig of each region in the partition region
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                    return ambiguous_regions, negative_regions
             
                
            vertices=np.array(vertices) 
            num_unique_vertices=len(np.unique(vertices,axis=0))

            p1=pc.qhull(vertices)                   
            # Compute H-representations of  p1 
            A1=p1.A
            b1=p1.b      
            if p1.volume==0:
                
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    # Compute Aambig and bambig of each region in the partition region
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                    return ambiguous_regions, negative_regions
                
            else:
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    negative_regions.append([{'A':p1.A,'b':p1.b}]) 
                    
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p1)       
                    p4=polype.diff(p1) 
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])             
                    return ambiguous_regions, negative_regions
        
        
        
        
        
        
        
        
    # ========================================================
    # Partition the region (Pos+Ambig) based on the polynomial sign
    # ========================================================  
    def Ppartition_inequality_regions(self,inequality_index, pregion): 
        num_samples=3
        poly=self.poly_inequality_coeffs[inequality_index]
        
        # Initialization of the pos, ambig, regions
        positive_regions        = []
        ambiguous_regions       = [] 
        # Transform pregion into polytope format
        Areg=pregion[0]['A']
        breg=pregion[0]['b']
        polype=pc.Polytope(Areg, breg) 
        
        
        # Compute the middle point in the polytope region
        rb,mid_point=pc.cheby_ball(polype)
     
        # Compute the hessian Matrix of Poly at mid_point
        Hess=self.Hessian(poly,mid_point)
    
        # Compute the gradient vector of Poly at mid_point
        Gradi=self.Gradient(poly,mid_point)
        
        # Compute the remainder for the 2nd order Taylor overapproximation
        Rem2=self.remainder2cst(poly,pregion,mid_point,Gradi,Hess)
        
        # Check if the Taylor overapprox is two sheet or one sheet
        if self.num_pos_eig(Hess)==1: # There is two sheets
            # Compute the hyperplane (As,bs) that will separate the two sheets
              
            #  Compute the center of the hyperbola
            coefcen=Hess.dot(mid_point)-Gradi
            cen=np.linalg.solve(Hess,coefcen)
            # Compute the principal axis As
            eigval,eigvec=np.linalg.eig(Hess)
            # Compute the index of the eigenvalue > 0
            indexaux=np.nonzero(eigval> 0)
            index=indexaux[0][0]
            # Compute As and bs
            As=eigvec[index,:]
            bs=As.dot(cen)
             
            # Compute 2n vertices (2n faces) of the two polytope that under-approximate the negative regions if they exist      
            # Construct 2n template vectors 
            Tem1=np.identity(self.num_vars)
            Tem2=-np.identity(self.num_vars)
            Tem=np.insert(Tem1,self.num_vars,Tem2,axis=0)
        
            # Compute the vertices of the two polytopes  
            vertices1=[]
            vertices2=[]
            for i in range(2*(self.num_vars)):

                v1=self.Ver_tang_two_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],As,bs,'P')
                v2=self.Ver_tang_two_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],-As,-bs,'P')
                v1=list(v1)
                v2=list(v2)
            
            
                # if the vertices exist==> Neg Reg exist==> Save it
               # print(v1,v2)
                if (len(v1)!=0) and (len(v2)!=0):
                    vertices1.append(v1)
                    vertices2.append(v2)
                elif (len(v1)!=0):  
                    vertices1.append(v1)
                elif (len(v2)!=0):  
                    vertices2.append(v2)    
                else: # if the vertices  does not exist==> Pos Reg does not exist==> Partition Ambiguous Region
                    # create a polytope that is inside pregion formed by a center and n vertices
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, positive_regions
        
                    
            vertices1=np.array(vertices1)
            vertices2=np.array(vertices2)
            
            
            
            if (len(vertices1)!=0) and (len(vertices2)!=0): 
                # Compute the two polytopes p1 and p2 using the H-representations
                p1 = pc.qhull(vertices1)
                p2 = pc.qhull(vertices2)          
                # Compute H-representations of  p1 and p2
                A1=p1.A
                b1=p1.b
                A2=p2.A
                b2=p2.b               
                # Compute the polytope p3 the union of p1 and p2 (it presents the negative region)
                if ((p1.volume==0) and (p2.volume==0)):
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, positive_regions
                    
                    
                elif ((p1.volume!=0) and (p2.volume==0)):
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1)  
                    p1=pc.box2poly(boxunrec1)
                    positive_regions.append([{'A':p1.A,'b':p1.b}])
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p1)
                    p4=polype.diff(p1) 
                    
                elif ((p1.volume==0) and (p2.volume!=0)):
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    positive_regions.append([{'A':p2.A,'b':p2.b}])
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p2)
                    p4=polype.diff(p2)
                    
                elif ((p1.volume!=0) and (p2.volume!=0)):
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    positive_regions.append([{'A':p1.A,'b':p1.b}])
                    
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec1=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    positive_regions.append([{'A':p2.A,'b':p2.b}])
                    p3=p2.union(p1)             
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p3)
                    p4=polype.diff(p3) 
                    
                    
 
                # Compute the H-repren of each polytope in p4
                if len(p4)==0:
                    if p4.volume==0:
                        ambiguous_regions=ambiguous_regions+[]
                    else:
                        ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                else:
                    for polytope in p4:
                        if polytope.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                
                return ambiguous_regions, positive_regions
            
            elif (len(vertices1)!=0) and (len(vertices2)==0):           
                # Compute the polytope p1 
                p1= pc.qhull(vertices1) 
                # Compute H-representations of  p1 
                A1=p1.A
                b1=p1.b   
                
                if p1.volume==0:
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, positive_regions
                    
                    
                    
                else:
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    positive_regions.append([{'A':p1.A,'b':p1.b}])                    
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p1)
                    p4=polype.diff(p1)     
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                                
                    return ambiguous_regions, positive_regions

                
            elif (len(vertices1)==0) and (len(vertices2)!=0):     
                # Compute the polytope p2 
                p2= pc.qhull(vertices2) 
                # Compute H-representations of  p2 
                A2=p2.A
                b2=p2.b   
                
                if p2.volume==0:
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])  
                                
                    return ambiguous_regions, positive_regions
                    
                    
                    
                else:
                    # obatin the underapprox rect 2
                    boxunrec2=self.RecinPolytope(A2,b2)
                    boxunrec2=np.append(boxunrec2[0], boxunrec2[1], axis=1) 
                    p2=pc.box2poly(boxunrec2)
                    positive_regions.append([{'A':p2.A,'b':p2.b}])                    
                    # Compute the polytope p4 which presents the ambiguous region (pregion-p2)
                    p4=polype.diff(p2)     
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}]) 
                                
                    return ambiguous_regions, positive_regions
 
 


        else: # There is only one sheet  
            # Compute 2n vertices (2n faces) of the polytope that under-approximate the negative region if it exists         
            # Construct 2n template vectors 
            Tem1=np.identity(self.num_vars)
            Tem2=-np.identity(self.num_vars)
            Tem=np.insert(Tem1,self.num_vars,Tem2,axis=0)
        
            # Compute the vertices   
            vertices=[]
            for i in range((self.num_vars)+1):
                # print('kkkkkkkkkkkkkk3'+str(i))

                v=self.Ver_tang_one_sheet(poly,pregion,mid_point,Hess,Gradi,Rem2,Tem[i,:],'N')
            
            
                # if the vertice exist==> Neg Reg exist==> Save it
                if len(v)!=0:
                    if v in vertices:
                        p1,p2=self.Part_polype(polype)
                        pambig=p1.union(p2)
                        # Compute Aambig and bambig of each region in the partition region
                        if len(pambig)==0:
                            if pambig.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])
    
                            
                        else:
                            for polytope in pambig:
                                if polytope.volume==0:
                                    ambiguous_regions=ambiguous_regions+[]
                                else:
                                    ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                        return ambiguous_regions, positive_regions
                    
                    else:
                        vertices.append(v)
                else: # if the vertice  does not exist==> Neg Reg does not exist==> Partition Ambiguous Region
                    # create a polytope that is inside pregion formed by a center and n vertices
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    # Compute Aambig and bambig of each region in the partition region
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                    return ambiguous_regions, positive_regions
             
                
 
            vertices=np.array(vertices)    

            p1=pc.qhull(vertices) 
            A1=p1.A
            b1=p1.b      
            if p1.volume==0:
                
                    p1,p2=self.Part_polype(polype)
                    pambig=p1.union(p2)
                    # Compute Aambig and bambig of each region in the partition region
                    if len(pambig)==0:
                        if pambig.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':pambig.A,'b':pambig.b}])

                        
                    else:
                        for polytope in pambig:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])                   
                    return ambiguous_regions, positive_regions
                
            else:
                    # obatin the underapprox rect 1
                    boxunrec1=self.RecinPolytope(A1,b1)
                    boxunrec1=np.append(boxunrec1[0], boxunrec1[1], axis=1) 
                    p1=pc.box2poly(boxunrec1)
                    positive_regions.append([{'A':p1.A,'b':p1.b}])       
                    p4=polype.diff(p1) 
                    # Compute the H-repren of each polytope in p4
                    if len(p4)==0:
                        if p4.volume==0:
                            ambiguous_regions=ambiguous_regions+[]
                        else:
                            ambiguous_regions.append([{'A':p4.A,'b':p4.b}]) 
                    else:
                        for polytope in p4:
                            if polytope.volume==0:
                                ambiguous_regions=ambiguous_regions+[]
                            else:
                                ambiguous_regions.append([{'A':polytope.A,'b':polytope.b}])             
                    return ambiguous_regions, positive_regions
            
            
        
    
    
    
    # ========================================================
    #   Use Z3 to Solve many multivariable Polynomial Constraints
    # ========================================================
     
    def solveSMT_many_multivars(self,polys, pregion):
        solver = z3.Solver()
        x = z3.Reals(self.strVari(self.num_vars))
        
        ccc=(pregion[0]['A'])*x
        cc=np.sum(ccc,axis=1)
        c=cc-pregion[0]['b']

 
   


        solver.add(x[0]<=3.0)
        solver.add(x[0]>=-1.0)

        solver.add(x[1]<=3.0)
        solver.add(x[1]>=-1.0)
        

        polycs=[]
        for poly in polys:
            poly_constraint = 0
            for monomial_counter in range(0,len(poly)):
                coeff = poly[monomial_counter]['coeff']
                vars  = poly[monomial_counter]['vars']
                product = coeff

                for var_counter in range(len(vars)):
                    power = vars[var_counter]['power']
                    var   = x[var_counter]  
                    product = product * (var**power)
                poly_constraint = poly_constraint + product
            solver.add(poly_constraint < 0)
            polycs.append(poly_constraint)
        
 
        if solver.check() == sat:
            model = solver.model()
            return model


        return None       
    
    
    
    # ========================================================
    #   Use Yices to Solve many multivariable 
    #          Polynomial Constraints
    # ========================================================
     
    def Yicesmany_multivars(self,polys, box, q):
    
        cfg = Config()
        cfg.default_config_for_logic('QF_NRA')
        ctx = Context(cfg)
        real_t = Types.real_type()
        x0 = Terms.new_uninterpreted_term(real_t, 'x0')
        x1 = Terms.new_uninterpreted_term(real_t, 'x1')
        x2 = Terms.new_uninterpreted_term(real_t, 'x2')
        x3 = Terms.new_uninterpreted_term(real_t, 'x3')
        x4 = Terms.new_uninterpreted_term(real_t, 'x4')
        x5 = Terms.new_uninterpreted_term(real_t, 'x5')
        x6 = Terms.new_uninterpreted_term(real_t, 'x6')
        x7 = Terms.new_uninterpreted_term(real_t, 'x7')
        x8 = Terms.new_uninterpreted_term(real_t, 'x8')
        x9 = Terms.new_uninterpreted_term(real_t, 'x9')
        x10 = Terms.new_uninterpreted_term(real_t, 'x10')
        x11 = Terms.new_uninterpreted_term(real_t, 'x11')
        x12 = Terms.new_uninterpreted_term(real_t, 'x12')
        x13 = Terms.new_uninterpreted_term(real_t, 'x13')
        x14 = Terms.new_uninterpreted_term(real_t, 'x14')
        x15 = Terms.new_uninterpreted_term(real_t, 'x15')
        x16 = Terms.new_uninterpreted_term(real_t, 'x16')
        x17 = Terms.new_uninterpreted_term(real_t, 'x17')
        x18 = Terms.new_uninterpreted_term(real_t, 'x18')
        x19 = Terms.new_uninterpreted_term(real_t, 'x19')
        x20 = Terms.new_uninterpreted_term(real_t, 'x20')
        x21 = Terms.new_uninterpreted_term(real_t, 'x21')
        x22 = Terms.new_uninterpreted_term(real_t, 'x22')
        x23 = Terms.new_uninterpreted_term(real_t, 'x23')
        x24 = Terms.new_uninterpreted_term(real_t, 'x24')
        x25 = Terms.new_uninterpreted_term(real_t, 'x25')
        x26 = Terms.new_uninterpreted_term(real_t, 'x26')
        x27 = Terms.new_uninterpreted_term(real_t, 'x27')
        x28 = Terms.new_uninterpreted_term(real_t, 'x28')
        x29 = Terms.new_uninterpreted_term(real_t, 'x29')
        x30 = Terms.new_uninterpreted_term(real_t, 'x30')
        x31 = Terms.new_uninterpreted_term(real_t, 'x31')
        x32 = Terms.new_uninterpreted_term(real_t, 'x32')

        fmlaliststr=[]
        fmlalist=[]
        i=0
        for poly in polys:
            res=self.fmlapoly(poly,self.type_regions[i])
            fmlaliststr.append(res)
            i=i+1
            
        fmlaliststr.append(self.fmlabounds2(box))    
        
        for i in range(len(fmlaliststr)):
            fmla=Terms.parse_term(fmlaliststr[i])
            fmlalist.append(fmla)
        
        ctx.assert_formulas(fmlalist)
        status = ctx.check_context()
        
        if status == Status.SAT:
            model = Model.from_context(ctx, 1)
            model_string = model.to_string(80, 100, 0)
            sol=np.array([model.get_value(x0),model.get_value(x1),model.get_value(x2),model.get_value(x3),model.get_value(x4),model.get_value(x5),model.get_value(x6),model.get_value(x7),model.get_value(x8)])
            q.put(sol)
            return 'SAT'
        
        else:
            return 'UNSAT'
        
        


# ========================================================
#   Function to output the n^th dimenstion hypercube
#      with edge limited between xmin and xmax
# ========================================================
def hypercube(n, xmin, xmax):
    box=[]
    for i in range(n):
        box.append([xmin,xmax])
    
    return box         
        
if __name__ == "__main__":

    num_vars = 2
    
    x_min = -1.0
    x_max = 1.0
    
    
    box=np.array(hypercube(num_vars, x_min,x_max))
    polype=pc.box2poly(box)
    boxx=pc.bounding_box(polype)
    pregion=[[{'A':polype.A,'b':polype.b}]]
    solver = PolyInequalitySolver(num_vars, pregion)
    
    # poly = 4 x^2 + 3 y^2 + 2
    poly = [
    {'coeff':4,      'vars':[{'power':2},{'power':0}]},
    {'coeff':3,    'vars':[{'power':0},{'power':2}]},
    {'coeff':2,    'vars':[{'power':0},{'power':0}]}
    ]
    
    

    
    solver.addPolyInequalityConstraint(poly)
    
    start_time = time.time()
    res=solver.solve()   

    print(res)     
        
        
        

    
        
        
    