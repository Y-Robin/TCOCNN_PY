"""Teaching cells for automatic FESR, six-source transfer and occlusion experiments."""

def fesr_cells(md,code,setup):
    return [md('''
# 09 - Automatische FESR-Pipeline: Aceton [ppb]

Die Anzahl und Grenzen der Signalabschnitte werden jetzt **gelernt**, nicht vorgegeben.
`AutomaticSegments.fit` berechnet den Median der Trainingszyklen und unterteilt ihn so lange,
wie der Gewinn an linearer Rekonstruktion den BIC-artigen Komplexitaetspreis uebersteigt.
Die Mindestlaenge von acht Samples ist eine Aufloesungsgrenze, keine Segmentzahl.

Das ist ein greedy Verfahren mit einem automatischen Abbruchkriterium, keine Garantie einer global
optimalen Segmentierung. Mittelwert und Steigung jedes gelernten Abschnitts bilden die Merkmale.
Die Grenzen bleiben bei `transform` fest: neue Messungen werden nicht separat neu segmentiert.
'''),code(setup+'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from dataset_pipeline import prepare_splits
from automatic_fesr import AutomaticSegments, FractionSelector
splits=prepare_splits('stored'); GAS='acetone'
X=splits['train']['X']; y=splits['train']['targets'][GAS]; groups=splits['train']['targets']['range']
extractor=AutomaticSegments().fit(X)
features=extractor.transform(X)
print('Automatisch gelernte Segmente:',extractor.n_segments_,'; Merkmale:',features.shape[1])
print('Train:',len(np.unique(groups)),'UGMs /',len(y),'Zyklen')
'''),md('''
## Automatische Aufteilung und Abbruch

Orange zeigt die lineare Rekonstruktion in den gelernten Abschnitten. Die rechte Grafik zeigt den
Komplexitaetswert auf dem akzeptierten Suchpfad. Die Segmentzahl wird in **jedem CV-Trainingsfold neu
gelernt**. Weder Validierungs- noch Testzyklen bestimmen den Median oder die Abschnittsgrenzen.
'''),code('''
fig,axes=plt.subplots(1,2,figsize=(15,5)); time=np.arange(X.shape[-1])/10
axes[0].plot(time,extractor.reference_,label='Trainingsmedian')
for i,(a,b) in enumerate(zip(extractor.boundaries_[:-1],extractor.boundaries_[1:])):
    t=time[a:b]; centered=t-t.mean(); values=extractor.reference_[a:b]
    fitted=values.mean()+centered*(values@centered)/(centered@centered)
    axes[0].plot(t,fitted,color='tab:orange')
axes[0].set(title=f'{extractor.n_segments_} automatisch gelernte Abschnitte',xlabel='Zeit [s]',ylabel='Gespeicherter Signalwert')
path=np.asarray(extractor.path_); axes[1].plot(path[:,0],path[:,1],marker='.')
axes[1].set(xlabel='Akzeptierte Segmentzahl',ylabel='BIC-artiger Komplexitaetswert',title='Stopp ohne vorgegebenes n_seg')
plt.tight_layout(); plt.show()
'''),md('''
## sklearn-Pipeline: alles innerhalb des Trainingsfolds lernen

`AutomaticSegments -> StandardScaler -> FractionSelector -> PLSRegression`.
Da die Zahl der Merkmale vom Fold abhaengt, waehlt die Feature Selection einen Anteil statt einer
festen Anzahl. Mindestens zwei Merkmale bleiben fuer die getesteten ein oder zwei PLS-Komponenten.
Pearson und RFE/Ridge werden ueber gruppierte Kreuzvalidierung auf Train verglichen; die separate
Validierung bestimmt die finale Variante. Aceton ist in allen Metriken in ppb angegeben.
'''),code('''
pipeline=Pipeline([('features',AutomaticSegments()),('scale',StandardScaler()),
                   ('select',FractionSelector()),('regression',PLSRegression(scale=False))])
searches={}; predictions={}; rows=[]
for method in ['pearson','rfe']:
    grid={'select__method':[method],'select__fraction':[.25,.5,1.], 'regression__n_components':[1,2]}
    search=GridSearchCV(pipeline,grid,cv=GroupKFold(3),scoring='neg_root_mean_squared_error',n_jobs=1,error_score='raise')
    search.fit(X,y,groups=groups); searches[method]=search
    predictions[method]=search.predict(splits['val']['X']).ravel()
    rows.append({'Methode':method,'CV_RMSE_ppb':-search.best_score_,
                 'Segmente':search.best_estimator_.named_steps['features'].n_segments_,
                 **regression_metrics(splits['val']['targets'][GAS],predictions[method])})
    print(method,search.best_params_)
show_results(rows)
fold_rows=[]
for fold,(train_idx,_) in enumerate(GroupKFold(3).split(X,y,groups),1):
    learned=AutomaticSegments().fit(X[train_idx])
    fold_rows.append({'Fold':fold,'Train_UGMs':len(np.unique(groups[train_idx])),'Segmente':learned.n_segments_})
show_results(fold_rows)
plot_comparison(splits['val']['targets'][GAS],predictions,'Aceton [ppb]: separate Validierung')
'''),code('''
winner=min(rows,key=lambda row:row['RMSE_ppb'])['Methode']; fitted=searches[winner].best_estimator_
fe=fitted.named_steps['features']; selected=fitted.named_steps['select'].indices_
plt.figure(figsize=(14,4)); plt.plot(time,X[0,0])
for segment in np.unique(selected//2):
    a,b=fe.boundaries_[segment:segment+2]; plt.axvspan(a/10,b/10,color='tab:orange',alpha=.3)
plt.title('Aceton: automatisch gelernte und ausgewaehlte Signalbereiche'); plt.xlabel('Zeit [s]'); plt.ylabel('Signal')
plt.tight_layout(); plt.show()
print('Gewaehlte Merkmale:',fe.get_feature_names_out()[selected].tolist())
final=[]
for name in ['test','test_extra']:
    truth=splits[name]['targets'][GAS]; pred=fitted.predict(splits[name]['X']).ravel()
    plot_comparison(truth,{winner:pred,'Trainingsmittelwert':np.full(len(truth),y.mean())},'Aceton: '+name)
    final.append({'Split':name,**regression_metrics(truth,pred)})
show_results(final)
''')]


def transfer_cells(md,code,setup):
    return [md('''
# 01 - Acetone calibration: global model from six sensors, transfer to the seventh

**Gas: acetone; all concentrations and errors are in ppb.** Each measurement is one full 1440-sample
cycle of channel 0. Six devices train a shared source model; the seventh device is the final target.
We concatenate **observations**, not sensor channels: input shape stays `(n, 1, 1440, 1)`.
Calibration 1 is fixed. The source model is trained here from random initialization and saved explicitly.

There are three separate decisions:
1. Select source architecture using only the six source sensors' grouped validation data.
2. Select transfer learning rate, epoch, DS regularization and PDS window using six internal
   leave-one-sensor-out experiments (train on five, adapt to the sixth).
3. Reuse those choices on the seventh sensor. Its validation plots are diagnostic; they do not choose
   transfer learning rate, epoch or checkpoint. Its test UGMs never enter fitting or selection.
'''),code(setup+'''
import torch
from day4_utils import (load_sensor_domains,source_scaler,scale_domain,transfer_subset,
    fit_network,predict_ppb,save_checkpoint,DEFAULT_PARAMS)
from transfer_workflow import (GAS,COLORS,METHODS,metrics,group_values,pool_domains,joint_training,
    copy_state,paired_master,DirectStandardization,PiecewiseStandardization,mapped_predictions,
    choose_settings,plot_predictions,save_json,run_transfer_search)
CHANNEL=0; CALIBRATION=1; SEED=42; SOURCE_EPOCHS=100
BUDGETS=[10,40]; LEARNING_RATES=[1e-5,1e-4,1e-3]; EPOCH_GRID=[20,40,60]
raw,audit=load_sensor_domains(CHANNEL,CALIBRATION)
sensors=list(raw); SOURCES=sensors[:6]; TARGET=sensors[6]; MASTER=SOURCES[0]
output=ROOT/'artifacts'/'seminar_day4_global6'; output.mkdir(parents=True,exist_ok=True)
print('Gas:',GAS,'[ppb]'); print('Source devices:',SOURCES); print('Final target:',TARGET,'; DS/PDS master:',MASTER)
show_results([{'stage':'Source training','devices':6,'unique_UGMs':audit['groups']['train'],
               'sensor_UGM_observations':6*audit['groups']['train'],'cycles':6*audit['rows']['train']},
              {'stage':'Source validation','devices':6,'unique_UGMs':audit['groups']['val'],
               'sensor_UGM_observations':6*audit['groups']['val'],'cycles':6*audit['rows']['val']},
              {'stage':'Target diagnostic validation','devices':1,'unique_UGMs':audit['groups']['val'],
               'sensor_UGM_observations':audit['groups']['val'],'cycles':audit['rows']['val']},
              {'stage':'Target final test','devices':1,'unique_UGMs':audit['groups']['test'],
               'sensor_UGM_observations':audit['groups']['test'],'cycles':audit['rows']['test']}])
'''),md('''
## Data budgets, signal shift, and metric definitions

The same 137 training mixtures are observed by six devices: **822 sensor-UGM observations are still
137 unique gas mixtures**, not 822 independent mixtures. Source validation contains 19 other mixtures
per source. The target's 40 test mixtures are absent from training on every device.
Target training budgets are 10 and 40 complete UGMs; the table below reports their actual cycle counts.

Cycle RMSE weights every cycle. UGM RMSE first averages truth and prediction within each mixture,
then computes RMSE of these means. For pooled sources we keep sensor-UGM means separate, avoiding
cancellation between devices. All selection and UGM scatterplots use this explicitly named metric.
Repeated cycles can reduce noise in UGM means, so the two errors need not be equal.
'''),code('''
source_raw=pool_domains(raw,SOURCES); scaler=source_scaler(source_raw)
domains={sensor:scale_domain(raw[sensor],scaler) for sensor in sensors}
pooled=pool_domains(domains,SOURCES)
show_results([{'target':TARGET,'training_UGMs':budget,
               'training_cycles':len(transfer_subset(domains[TARGET],budget)['train']['y']),
               'diagnostic_val_UGMs':audit['groups']['val'],'test_UGMs':audit['groups']['test']} for budget in BUDGETS])
fig,axes=plt.subplots(1,2,figsize=(15,4))
for sensor in sensors:
    axes[0].plot(raw[sensor]['train']['X'][0,0,:,0],label=sensor,alpha=.8)
    axes[1].hist(domains[sensor]['train']['X_z'][:,0,::12,0].ravel(),bins=60,density=True,histtype='step',label=sensor)
axes[0].set(xlabel='Time sample',ylabel='Stored channel-0 signal',title='Acetone experiment: same cycle, seven devices')
axes[1].set(xlabel='Six-source training Z-score',ylabel='Density',title='Shared normalization; no target re-scaling')
for ax in axes: ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
'''),md('''
## Where the original model comes from, and which hyperparameters it uses

We train two declared v3 architectures for 100 epochs on the pooled **six-source training data**.
The source-validation sensor-UGM RMSE selects the architecture and checkpoint. This is a small source
architecture comparison, separate from the much larger Day 3 search on a different dataset.
AdamW uses weight decay 1e-4, batch size 64 and a learning-rate halving every 30 epochs.
Temporal convolutions use stride 1; `stride=4` below is the pooling factor. Global average pooling
makes the head independent of cycle length. The printed network shows every actual layer.
'''),code('''
profiles=[{**DEFAULT_PARAMS,'n_filter':32,'section_depth':3,'channel_growth':16},
          {**DEFAULT_PARAMS,'n_filter':48,'section_depth':3,'channel_growth':16}]
source_rows=[]; source=None; best=np.inf
for index,settings in enumerate(profiles):
    candidate,history,info=fit_network(pooled,settings,epochs=SOURCE_EPOCHS)
    pred=predict_ppb(candidate,pooled['val'],scaler)
    score=metrics(pooled['val']['y'],pred,pooled['val']['eval_groups'])
    source_rows.append({'profile':index,**settings,**info,**score})
    if score['UGM_RMSE_ppb']<best:
        best=score['UGM_RMSE_ppb']; source=candidate; source_history=history; source_info=info; params=settings
show_results([{k:row[k] for k in ['profile','n_filter','section_depth','channel_growth','best_epoch','cycle_RMSE_ppb','UGM_RMSE_ppb']} for row in source_rows])
print('Chosen source hyperparameters:',params); print(source.model)
source_state=copy_state(source)
source_reference=pooled['train']['X_z'].mean(axis=0,keepdims=True,dtype=np.float64).astype(np.float32)
save_checkpoint(output/'source_global6',source,scaler,
    {'params':params,'gas':GAS,'sources':SOURCES,'target':TARGET,'channel':CHANNEL,'calibration':CALIBRATION,
     'source_train_UGMs':audit['groups']['train'],'source_train_cycles':len(pooled['train']['y']),
     'selected_epoch':source_info['best_epoch']},source_reference)
fig,ax=plt.subplots(figsize=(11,4))
ax.plot(np.arange(1,SOURCE_EPOCHS+1),np.sqrt(source_history['val_loss'])*scaler['y_std'],label='Cycle RMSE')
ax.plot(np.arange(1,SOURCE_EPOCHS+1),np.sqrt(source_history['val_group_mse'])*scaler['y_std'],label='Sensor-UGM RMSE')
ax.axvline(source_info['best_epoch'],color='black',linestyle=':',label='Restored source checkpoint')
ax.set(xlabel='Epoch',ylabel='Acetone validation RMSE [ppb]',title='Six-source model selection'); ax.legend()
plt.tight_layout(); plt.show()
'''),md('''
## Select transfer hyperparameters using only the six available source devices

Each of six folds holds out one source device as a **pseudo-target**. A fresh source model is trained
on the other five devices, with their training-only scaler and source-validation checkpoint. For each
10/40-UGM target budget we test three learning rates for scratch, head-only and full fine-tuning.
Checkpoints at epochs 20, 40 and 60 are scored on the pseudo-target's validation UGMs. We average UGM
RMSE equally over all six pseudo-targets and choose learning rate **and epoch separately per method and budget**.
Dropout, architecture, optimizer and weight decay remain the chosen source settings.

DS learns a regularized affine target-to-master mapping using the complete paired signals.
Sliding-window PDS predicts each master sample from a small neighbouring target window. Both use
exact paired measurement-row IDs from the selected training UGMs, never target validation/test signals
for fitting. DS alpha and PDS alpha/window are also chosen over the same six development folds.
The master is a member of the five training devices in each fold. The final master is printed above.
'''),code('''
search_rows=run_transfer_search(raw,SOURCES,params,BUDGETS,LEARNING_RATES,EPOCH_GRID,SOURCE_EPOCHS,output)
assert all(row['heldout']!=TARGET and TARGET not in row['train_sensors'].split(',') for row in search_rows)
chosen={(method,budget):choose_settings(search_rows,method,budget)
        for budget in BUDGETS for method in ['Scratch','Head-only','Fine-tune','DS','PDS']}
selection_rows=[]
for (method,budget),setting in chosen.items():
    selection_rows.append({'method':method,'training_UGMs':budget,
        'lr':f"{setting['lr']:.1e}" if setting['lr'] is not None else '-',
        'epochs':setting['epoch'] if setting['epoch'] is not None else '-',
        'DS_PDS_alpha':setting['alpha'] if setting['alpha'] is not None else '-',
        'PDS_window_samples':2*setting['radius']+1 if setting['radius'] is not None else '-',
        'six_sensor_mean_UGM_RMSE_ppb':setting['development_RMSE']})
show_results(selection_rows)
save_json(output/'chosen_transfer_settings.json',selection_rows)
fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
for ax,budget in zip(axes,BUDGETS):
    for method in ['Scratch','Head-only','Fine-tune']:
        values=[]
        for lr in LEARNING_RATES:
            values.append(min(np.mean([r['UGM_RMSE_ppb'] for r in search_rows if r['method']==method and r['budget']==budget and r.get('lr')==lr and r.get('epoch')==epoch]) for epoch in EPOCH_GRID))
        ax.plot(LEARNING_RATES,values,marker='o',color=COLORS[method],label=method)
    ax.set(xscale='log',xlabel='Transfer learning rate',ylabel='Six-fold mean UGM RMSE [ppb]',title=f'Acetone: {budget} adaptation UGMs'); ax.legend()
plt.tight_layout(); plt.show()
'''),
md('''
## Final seventh sensor: methods and model selection boundaries

- **Global-6:** the pretrained six-source model applied unchanged.
- **Global + target:** train a joint model from scratch on the six sources plus the selected target
  training rows. All rows have equal weight; there is no hidden target oversampling. The checkpoint
  is selected on the six-source validation data only.
- **DS / PDS:** map paired target signals to the printed master device, then apply Global-6.
- **Scratch:** train only on the selected target rows, with development-selected learning rate/epoch.
- **Head-only:** adapt the dense head; freeze feature weights and BatchNorm running statistics.
- **Fine-tune:** adapt all weights and running statistics.

For the last three methods the final epoch was selected on the six development devices. **No best
seventh-sensor validation checkpoint is restored.** Curves and scatterplots therefore refer to the
same saved state. Source pretraining and development search are additional costs. Diagnostic target
validation requires 19 extra labelled UGMs, but those labels do not choose these transfer models.
'''),code('''
validation_predictions={}; test_predictions={}; histories={}; model_info={}; result_rows=[]
for budget in BUDGETS:
    subset=transfer_subset(domains[TARGET],budget); paired=paired_master(domains[MASTER],subset)
    for method in METHODS:
        print('Final target',TARGET,'acetone',budget,'training UGMs:',method,flush=True)
        key=(budget,method); history={}; info={}
        if method=='Global-6':
            model=source
            val=predict_ppb(model,domains[TARGET]['val'],scaler); test=predict_ppb(model,domains[TARGET]['test'],scaler)
        elif method=='Global + target':
            model,history,info=fit_network(joint_training(pooled,subset),params,epochs=SOURCE_EPOCHS)
            val=predict_ppb(model,domains[TARGET]['val'],scaler); test=predict_ppb(model,domains[TARGET]['test'],scaler)
        elif method in ['DS','PDS']:
            setting=chosen[(method,budget)]
            adapter=DirectStandardization(setting['alpha']) if method=='DS' else PiecewiseStandardization(setting['alpha'],setting['radius'])
            adapter.fit(subset['train']['X_z'],paired)
            val=mapped_predictions(source,adapter,domains[TARGET]['val'],scaler)
            test=mapped_predictions(source,adapter,domains[TARGET]['test'],scaler)
        else:
            setting=chosen[(method,budget)]; settings={**params,'initial_learning_rate':setting['lr']}
            model,history,info=fit_network(subset,settings,epochs=setting['epoch'],
                initial=None if method=='Scratch' else source_state,head_only=method=='Head-only',restore_best=False)
            val=predict_ppb(model,domains[TARGET]['val'],scaler); test=predict_ppb(model,domains[TARGET]['test'],scaler)
            score=metrics(domains[TARGET]['val']['y'],val,domains[TARGET]['val']['groups'])
            assert np.isclose(score['UGM_RMSE_ppb'],np.sqrt(history['val_group_mse'][-1])*scaler['y_std'],rtol=2e-4,atol=.01)
            if method=='Fine-tune' and budget==BUDGETS[-1]:
                reference=subset['train']['X_z'].mean(axis=0,keepdims=True,dtype=np.float64).astype(np.float32)
                save_checkpoint(output/'target_finetuned',model,scaler,
                    {'params':settings,'gas':GAS,'sources':SOURCES,'target':TARGET,'channel':CHANNEL,'calibration':CALIBRATION,
                     'budget_UGMs':budget,'adaptation_epochs':setting['epoch'],'seed':SEED,
                     'train_rows':subset['train']['rows'].tolist(),'selection':'six-source leave-one-sensor-out'},reference)
        validation_predictions[key]=val; test_predictions[key]=test; histories[key]=history; model_info[key]=info
        result_rows.append({'method':method,'training_UGMs':budget,**metrics(domains[TARGET]['val']['y'],val,domains[TARGET]['val']['groups'])})
show_results(result_rows)
'''),md('''
## Reconcile learning curves, colours, and UGM scatterplots

Each method has one fixed colour in every plot. Only target-trained methods appear in the target
validation learning curves: a joint model's source-validation curve would describe a different
population and is deliberately not mixed in. Cycle and UGM errors are both RMSE in ppb. A dot marks
the final development-selected epoch, which must equal the scatterplot's diagnostic validation error.
'''),code('''
for budget in BUDGETS:
    fig,axes=plt.subplots(1,2,figsize=(14,4))
    for method in ['Scratch','Head-only','Fine-tune']:
        history=histories[(budget,method)]; epochs=np.arange(1,len(history['val_loss'])+1)
        for ax,key in zip(axes,['val_loss','val_group_mse']):
            values=np.sqrt(history[key])*scaler['y_std']
            ax.plot(epochs,values,color=COLORS[method],label=method)
            ax.scatter(epochs[-1],values[-1],color=COLORS[method],s=55)
    for ax,label in zip(axes,['Cycle RMSE','UGM-mean RMSE']):
        ax.set(xlabel='Epoch',ylabel=f'Acetone {label} [ppb]',title=f'{TARGET}: {budget} train UGMs; 19 validation UGMs'); ax.legend()
    plt.tight_layout(); plt.show()
    plot_predictions(domains[TARGET]['val'],{method:validation_predictions[(budget,method)] for method in METHODS},
        f'Acetone validation, target {TARGET}, {budget} train UGMs',grouped=True)
'''),md('''
## Frozen final evaluation

Both budgets and all seven methods were specified before looking at the final target test. We report
individual-cycle errors and UGM-mean errors, and show the same 40 test UGMs in the scatterplots. A lower
UGM error does not contradict a larger cycle error: averaging and weighting differ. No target-test
result selects a learning rate, epoch or method.
'''),code('''
final=[]
for budget in BUDGETS:
    preds={method:test_predictions[(budget,method)] for method in METHODS}
    plot_predictions(domains[TARGET]['test'],preds,f'Acetone TEST, {TARGET}, {budget} training UGMs',grouped=True)
    for method,pred in preds.items(): final.append({'method':method,'training_UGMs':budget,**metrics(domains[TARGET]['test']['y'],pred,domains[TARGET]['test']['groups'])})
plot_predictions(domains[TARGET]['test'],{method:test_predictions[(BUDGETS[-1],method)] for method in METHODS},
    f'Acetone TEST, {TARGET}, {BUDGETS[-1]} training UGMs',grouped=False)
show_results(final); save_json(output/'final_metrics.json',final)
print('Original six-source model:',output/'source_global6')
print('Fine-tuned model used for occlusion:',output/'target_finetuned')
''')]


def occlusion_cells(md,code,setup):
    return [md('''
# 02 - Acetone: occlusion and building up a cycle from Top 1 to Top 12

This notebook uses **occlusion only**. We explain the fine-tuned seventh-sensor model from Notebook 01.
A cycle consists of 12 consecutive high/low pairs: 50 high samples followed by 70 low samples.
A pair is kept intact when selecting Top 1, Top 2, etc. Channel 0 and acetone [ppb] remain unchanged.

We rank pairs by mean absolute prediction change when removing them from the **40 adaptation-training
UGMs**. No test values or test labels rank pairs. Two experiments then use the same fixed ranking:
1. Keep Top k pairs and replace the rest by the adaptation-training reference cycle, without retraining.
2. Actually concatenate only Top k pairs (in original temporal order) and adapt a fresh copy of the
   six-source model using these reduced inputs. This tests whether a model can use the retained information.

The second experiment uses the learning rate/epoch already selected on the six source devices.
Target validation (19 UGMs) selects k. All k settings are frozen before reporting the final 40-test-UGM curve.
'''),code(setup+'''
from day4_utils import load_checkpoint,load_sensor_domains,scale_domain,transfer_subset,fit_network,predict_ppb,occlusion
from transfer_workflow import metrics,plot_predictions,copy_state,save_json
output=ROOT/'artifacts'/'seminar_day4_global6'
model,scaler,metadata=load_checkpoint(output/'target_finetuned')
source,_,source_metadata=load_checkpoint(output/'source_global6')
raw,audit=load_sensor_domains(metadata['channel'],metadata['calibration'])
domain=scale_domain(raw[metadata['target']],scaler)
subset=transfer_subset(domain,metadata['budget_UGMs'],metadata['seed'])
assert np.array_equal(subset['train']['rows'],metadata['train_rows'])
reference=scaler['reference_cycle']; source_state=copy_state(source)
print('Gas:',metadata['gas'],'[ppb]; target:',metadata['target'])
print('Ranking/adaptation:',len(np.unique(subset['train']['groups'])),'UGMs /',len(subset['train']['y']),'cycles')
print('Validation:',audit['groups']['val'],'UGMs; test:',audit['groups']['test'],'UGMs')
print('Fixed transfer parameters:',metadata['params'],'epochs:',metadata['adaptation_epochs'])
'''),md('''
## Occlude high, low, and complete pairs

The plotted effect is `prediction(original) - prediction(replaced)` in ppb, not an error increase.
A positive value means replacement lowers the prediction. Absolute effects describe model dependence;
they do not prove chemical selectivity. Group-average effects prevent mixtures with more repeated cycles
from dominating the ranking. The reference contains only adaptation-training data.
'''),code('''
X=subset['train']['X_z']; original=model.predict(X).ravel()*float(scaler['y_std'])+float(scaler['y_mean'])
effects={kind:[] for kind in ['high','low','pair']}
for pair in range(12):
    for kind,a,b in [('high',pair*120,pair*120+50),('low',pair*120+50,(pair+1)*120),('pair',pair*120,(pair+1)*120)]:
        masked=X.copy(); masked[:,:,a:b]=reference[:,:,a:b]
        effects[kind].append(original-(model.predict(masked).ravel()*float(scaler['y_std'])+float(scaler['y_mean'])))
effects={kind:np.stack(value,axis=1) for kind,value in effects.items()}
groups=subset['train']['groups']; unique=np.unique(groups)
importance={kind:np.mean([np.abs(value[groups==group]).mean(axis=0) for group in unique],axis=0) for kind,value in effects.items()}
ranking=np.argsort(-importance['pair'],kind='stable')
show_results([{'rank':rank+1,'pair':int(pair+1),'high_samples':'50','low_samples':'70',
               'sample_start':int(pair*120),'sample_stop_exclusive':int((pair+1)*120),
               'mean_abs_occlusion_ppb':importance['pair'][pair]} for rank,pair in enumerate(ranking)])
fig,axes=plt.subplots(1,2,figsize=(15,5))
for kind in ['high','low','pair']: axes[0].plot(np.arange(1,13),importance[kind],marker='o',label=kind)
axes[0].set(xlabel='Original high/low pair',ylabel='Mean absolute prediction change [ppb]',title='Acetone: 40 training UGMs'); axes[0].legend()
limit=max(float(np.abs(effects['pair']).max()),1e-6)
im=axes[1].imshow(effects['pair'],aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
axes[1].set(xlabel='Pair index (zero-based)',ylabel='Adaptation cycle',title='Signed pair occlusion [ppb]'); fig.colorbar(im,ax=axes[1])
plt.tight_layout(); plt.show()
'''),md('''
## Top-k masking versus a reduced-input transfer model

Masking is a counterfactual query of the existing model. Reduced-input adaptation actually receives
only 120*k samples. Global average pooling permits reuse of the six-source weights at each length.
For each k we start from that **same source checkpoint**, not from the previous k or a model selected
on target test results. All 12 reduced models use the same target training UGMs and fixed transfer settings.

Times below are retained samples / 10 Hz (12 seconds per high/low pair), not a validated new physical
measurement cycle. Removing a heater phase can change subsequent dynamics; a physical optimum requires
new measurements with the changed heating protocol.
'''),code('''
rows=[]; val_predictions={}; test_predictions={}; masked_val={}; masked_test={}
for k in range(1,13):
    kept=np.sort(ranking[:k]); positions=np.concatenate([np.arange(pair*120,(pair+1)*120) for pair in kept])
    reduced={name:{**split,'X_z':split['X_z'][:,:,positions,:]} for name,split in subset.items()}
    reduced_model,_,info=fit_network(reduced,metadata['params'],epochs=metadata['adaptation_epochs'],
                                    initial=source_state,seed=metadata['seed'],restore_best=False)
    for name,cache,mask_cache in [('val',val_predictions,masked_val),('test',test_predictions,masked_test)]:
        cache[k]=predict_ppb(reduced_model,reduced[name],scaler)
        masked=np.broadcast_to(reference,domain[name]['X_z'].shape).copy()
        masked[:,:,positions,:]=domain[name]['X_z'][:,:,positions,:]
        mask_cache[k]=model.predict(masked).ravel()*float(scaler['y_std'])+float(scaler['y_mean'])
    score=metrics(domain['val']['y'],val_predictions[k],domain['val']['groups'])
    mask_score=metrics(domain['val']['y'],masked_val[k],domain['val']['groups'])
    rows.append({'k':k,'kept_pairs':','.join(str(int(p+1)) for p in kept),'samples':120*k,'nominal_seconds':12*k,
                 'retrained_val_UGM_RMSE_ppb':score['UGM_RMSE_ppb'],'masked_val_UGM_RMSE_ppb':mask_score['UGM_RMSE_ppb']})
show_results(rows)
best_k=min(rows,key=lambda row:row['retrained_val_UGM_RMSE_ppb'])['k']
print('Frozen validation-selected k:',best_k)
fig,ax=plt.subplots(figsize=(11,5))
ax.plot([r['nominal_seconds'] for r in rows],[r['retrained_val_UGM_RMSE_ppb'] for r in rows],marker='o',label='Reduced-input adaptation')
ax.plot([r['nominal_seconds'] for r in rows],[r['masked_val_UGM_RMSE_ppb'] for r in rows],marker='s',label='Masking, no retraining')
ax.axvline(12*best_k,color='black',linestyle=':',label='Validation optimum')
ax.set(xlabel='Retained samples / 10 Hz [s]',ylabel='Acetone validation UGM RMSE [ppb]',title='Top 1 to Top 12; 19 validation UGMs'); ax.legend()
plt.tight_layout(); plt.show()
'''),md('''
## Final test curve after freezing the ranking, settings, and chosen k

We report all predeclared k values to show how information accumulates; the selected optimum is still
the **validation** minimum. A smaller test error at another k must not silently replace that choice.
The scatterplots show Top 1, Top 2, the validation winner and all 12 pairs on the same held-out UGMs.
'''),code('''
final=[]
for k in range(1,13):
    score=metrics(domain['test']['y'],test_predictions[k],domain['test']['groups'])
    final.append({'k':k,'nominal_seconds':12*k,'chosen_on_validation':k==best_k,**score})
show_results(final)
fig,ax=plt.subplots(figsize=(11,5))
ax.plot([12*k for k in range(1,13)],[r['UGM_RMSE_ppb'] for r in final],marker='o',label='Test UGM RMSE')
ax.plot([12*k for k in range(1,13)],[r['cycle_RMSE_ppb'] for r in final],marker='s',label='Test cycle RMSE')
ax.axvline(12*best_k,color='black',linestyle=':',label='Validation-selected k')
ax.set(xlabel='Retained samples / 10 Hz [s]',ylabel='Acetone test RMSE [ppb]',title='40 unseen test UGMs; selection already frozen'); ax.legend()
plt.tight_layout(); plt.show()
selected=list(dict.fromkeys([1,2,best_k,12]))
plot_predictions(domain['test'],{f'Top {k} pairs':test_predictions[k] for k in selected},'Acetone: reduced-input test comparison',grouped=True)
save_json(output/'occlusion_cycle_buildup.json',{'ranking_zero_based':ranking.tolist(),'validation':rows,'selected_k':best_k,'test':final})
''')]
