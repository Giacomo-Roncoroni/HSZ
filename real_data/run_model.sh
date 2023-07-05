for j in 13 26 39 60
	do
		for k in 0.0125
		do 
			for i in {1..32} 
			do
				python generate_model.py $i 20 40 0.01 $j 20.04 16.32 $k 1 907e-9 7
	    			python -m gprMax models/model_$j$k$i.in -gpu
			done
		done
	done

