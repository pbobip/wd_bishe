<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { useTaskWorkflow } from '../composables/useTaskWorkflow'

const router = useRouter()
const workflow = useTaskWorkflow()
const resolving = ref(true)

onMounted(async () => {
  try {
    await workflow.initialize()
    await workflow.restoreDraftRunFromSession()
    if (workflow.activeRunId.value) {
      await router.replace(`/runs/${workflow.activeRunId.value}`)
      return
    }
    ElMessage.warning('请先在任务创建页开始主分割，再进入结果展示与后处理。')
  } catch {
    ElMessage.error('当前任务结果页打开失败')
  } finally {
    resolving.value = false
  }
})
</script>

<template>
  <section class="glass-card redirect-card">
    <div class="redirect-copy">
      <h2 class="section-title">结果展示与后处理</h2>
      <p class="section-subtitle">
        <template v-if="resolving">正在打开当前任务结果页…</template>
        <template v-else>当前没有可打开的任务，请先返回任务创建页完成图像登记并启动主分割。</template>
      </p>
    </div>
    <el-button type="primary" size="large" @click="router.push('/tasks/create')">返回任务创建</el-button>
  </section>
</template>

<style scoped>
.redirect-card {
  display: grid;
  gap: 16px;
  justify-items: start;
  padding: 22px;
}

.redirect-copy {
  display: grid;
  gap: 6px;
}
</style>
