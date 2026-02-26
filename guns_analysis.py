import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")

df = pd.DataFrame({"state":["Texas","California","Florida","New York","Illinois","Georgia","Texas","Ohio","Pennsylvania","Arizona","Texas","California","Florida","New York","Illinois","North Carolina","Michigan","Washington","Colorado","Tennessee"],"city":["Houston","Los Angeles","Miami","New York City","Chicago","Atlanta","Dallas","Columbus","Philadelphia","Phoenix","San Antonio","San Francisco","Orlando","Buffalo","Springfield","Charlotte","Detroit","Seattle","Denver","Nashville"],"n_killed":[0,1,0,2,1,0,1,0,2,0,0,1,0,1,0,2,1,0,1,0],"n_injured":[2,0,1,3,2,0,1,2,0,1,3,0,2,1,0,2,0,1,2,1],"incident_type":["Shooting","Shooting","Shooting","Shooting","Shooting","Accidental","Robbery","Shooting","Domestic","Robbery","Shooting","Suicide","Domestic","Robbery","Accidental","Shooting","Domestic","Robbery","Shooting","Shooting"],"age_group":["Adult","Adult","Teen","Adult","Adult","Child","Adult","Adult","Adult","Teen","Adult","Adult","Adult","Adult","Adult","Adult","Adult","Teen","Adult","Adult"],"lat":[29.76,34.05,25.79,40.71,41.85,33.75,32.78,39.96,39.95,33.45,29.42,37.77,28.54,42.88,39.78,35.22,42.33,47.61,39.74,36.17],"lon":[-95.37,-118.24,-80.13,-74.01,-87.65,-84.39,-96.80,-82.99,-75.16,-112.07,-98.49,-122.42,-81.38,-78.87,-89.65,-80.84,-83.05,-122.33,-104.98,-86.78],"month":[1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]})
df["cas"]=df["n_killed"]+df["n_injured"]

print("="*40,"PANDAS OPERATIONS","="*40)
print("\n1. Describe:\n",df[["n_killed","n_injured","cas"]].describe().round(2))
print("\n2. GroupBy State:\n",df.groupby("state")["cas"].sum().nlargest(3))
print("\n3. Pivot Table:\n",df.pivot_table("cas","age_group","incident_type","sum",fill_value=0))

print("\n"+"="*40,"NUMPY OPERATIONS","="*40)
k,inj,cas,lat,lon=df["n_killed"].values,df["n_injured"].values,df["cas"].values,df["lat"].values,df["lon"].values
df["z"]=(cas-cas.mean())/cas.std(); df["risk"]=k*2+inj
a=np.sin(np.radians(lat-41.88)/2)**2+np.cos(np.radians(lat))*np.cos(np.radians(41.88))*np.sin(np.radians(lon+87.63)/2)**2; df["d_chi"]=(6371*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))).round(1)
print(f"\n1. Stats — Mean:{np.mean(cas):.2f}  Std:{np.std(cas):.2f}  Percentiles(25,50,75):{np.percentile(cas,[25,50,75])}")
print("\n2. Outliers (|z|>1):\n",df[df["z"].abs()>1][["state","cas","z"]].round(2).to_string(index=False))
print("\n3. Top Risk Score:\n",df.nlargest(3,"risk")[["state","city","risk"]].to_string(index=False))
