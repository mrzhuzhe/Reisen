_config = {
    "Configuration": 
	{
		"numberOfStepsPerRenderUpdate": 1,
		"particleRadius": 0.01,
		
		"density0": 1000, 

        "domainEnd": [5.0, 3.0, 2.0]
    },
    "FluidBlocks": [
		{
            "objectId": 0,
			"start": [0.1, 0.1, 0.5],
			"end": [1.2, 2.9, 1.6],
			"translation": [0.2, 0.0, 0.2],
			"scale": [1, 1, 1],
			"velocity": [0.0, -1.0, 0.0],
			"density": 1000.0,
			"color": [50, 100, 200]
			
		}
	],
	"RigidBlocks": [
		{
            "objectId": 1,
			"start": [1.4, 0.1, 0.5],
			"end": [2.2, 0.5, 1.6],
			"translation": [0.2, 0.0, 0.2],
			"scale": [1, 1, 1],
			"velocity": [0.0, -1.0, 0.0],
			"density": 1000.0,
			"color": [50, 100, 200]
		}
	]
}