import { motion } from 'framer-motion';

export default function MiniMetric({ icon, label, value, onClick }) {
  return (
    <motion.button className="mini-metric" whileHover={{ y: -6, scale: 1.025 }} whileTap={{ scale: 0.98 }} onClick={onClick}>
      <span>{icon}</span>
      <strong>{value}</strong>
      <label>{label}</label>
    </motion.button>
  );
}
