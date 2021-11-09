for j in 10 15 20 25 30 35 40 45 50 55 60 
	do
		for k in 0.1 0.15 0.2 0.25 0.3 0.4
		do 
			for i in {1..25}
			do
				python generate_model.py $i 20 40 0.01 $j 15.5 12.98 $k 0 400e-9 7
	    			python -m gprMax models/model_$j$k$i.in -gpu
			done
		done
	done

