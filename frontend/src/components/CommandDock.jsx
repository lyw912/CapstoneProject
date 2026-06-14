import { motion } from 'framer-motion';
import {
  LoadingOutlined,
  PlayCircleOutlined,
  FileTextOutlined,
  MessageOutlined,
  CheckCircleOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

export default function CommandDock({ latest, task, onOpen, onRun, onReport, onFeedback }) {
  return (
    <div className="command-dock">
      <motion.button whileHover={{ y: -5 }} whileTap={{ scale: 0.98 }} className="dock-primary" onClick={onRun} disabled={task?.status === 'running'}>
        {task?.status === 'running' ? <LoadingOutlined /> : <PlayCircleOutlined />}
        <span>{task?.status === 'running' ? 'Running' : 'Run'}</span>
      </motion.button>
      <motion.button whileHover={{ y: -5 }} whileTap={{ scale: 0.98 }} onClick={onReport}><FileTextOutlined /><span>Draft</span></motion.button>
      <motion.button whileHover={{ y: -5 }} whileTap={{ scale: 0.98 }} onClick={onFeedback}><MessageOutlined /><span>Revise</span></motion.button>
      <motion.button whileHover={{ y: -5 }} whileTap={{ scale: 0.98 }} onClick={onOpen}>{latest ? <CheckCircleOutlined /> : <ExperimentOutlined />}<span>{latest ? 'Open' : 'Setup'}</span></motion.button>
    </div>
  );
}
