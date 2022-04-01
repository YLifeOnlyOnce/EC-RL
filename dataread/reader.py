from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
# 日志地址
logs_url = '../logstrain/events.out.tfevents.1648804625.Xiangbo-mbp.local.97405.0'
# 加载日志数据
ea = event_accumulator.EventAccumulator(logs_url)
ea.Reload()
print(ea.scalars.Keys())

val_energy = ea.scalars.Items('train_reward_energy')
print(len(val_energy))
print([(i.step,i.value) for i in val_energy])

fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(111)
train_reward_energy=ea.scalars.Items('train_reward_energy')
ax1.plot([i.step for i in train_reward_energy],[i.value for i in train_reward_energy],label='train_reward_energy')
ax1.set_xlim(0)
train_reward = ea.scalars.Items('train_reward')
ax1.plot([i.step for i in train_reward],[i.value for i in train_reward],label='train_reward')
ax1.set_xlabel("step")
ax1.set_ylabel("")

plt.legend(loc='lower right')
plt.show()