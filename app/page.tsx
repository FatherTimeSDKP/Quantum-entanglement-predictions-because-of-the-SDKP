"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Rocket, BarChart3, Target, Atom } from "lucide-react"

interface Particle {
  N: number
  S: number
  D: number
  v: number
  shape?: number
  dimension?: number
  number?: number
}

interface SpaceTime {
  x: number
  y: number
  z: number
  t: number
}

interface AnalysisResult {
  entangled: boolean
  confidence: number
  sdkpStrength: number
  sdnPattern: string
  sdnConfidence: number
  qccStrength: number
  spatialSeparation: number
}

class QuantumSDKPSimulator {
  private hbar = 1.054571817e-34
  private c = 299792458
  private alpha = 0.6
  private beta = 0.4
  private gamma = 1.5
  private delta = 0.8

  massEnergyScaling(N: number, S: number, D: number, v: number): number {
    const lorentzFactor = Math.sqrt(1 - (v / this.c) ** 2)
    const topologicalFactor = 1 + this.alpha * (N - 1) + this.beta * (S - 1) + this.gamma * (D - 1)
    return topologicalFactor * lorentzFactor
  }

  calculateSDKPDistance(p1: Particle, p2: Particle): number {
    return Math.sqrt((p1.N - p2.N) ** 2 + (p1.S - p2.S) ** 2 + (p1.D - p2.D) ** 2 + (p1.v - p2.v) ** 2)
  }

  entanglementCorrelationStrength(p1: Particle, p2: Particle): number {
    const distance = this.calculateSDKPDistance(p1, p2)
    return this.delta * Math.exp(-distance / (this.hbar * this.c))
  }

  detectSDNEntanglement(sdn1: number[], sdn2: number[]): { pattern: string; confidence: number } {
    const patterns = {
      symmetric: (s1: number, d1: number, n1: number) => [-s1, d1, n1],
      asymmetric: (s1: number, d1: number, n1: number) => [s1, -d1, n1],
      mixed: (s1: number, d1: number, n1: number) => [-s1, -d1, -n1],
      conserved: (s1: number, d1: number, n1: number) => [s1, d1, -n1],
    }

    let bestPattern = "symmetric"
    let maxScore = 0

    for (const [name, func] of Object.entries(patterns)) {
      const [es2, ed2, en2] = func(sdn1[0], sdn1[1], sdn1[2])
      const shapeMatch = 1 - Math.abs(sdn2[0] - es2) / 2
      const dimMatch = 1 - Math.abs(sdn2[1] - ed2) / Math.max(Math.abs(sdn2[1]), Math.abs(ed2), 1)
      const numMatch = 1 - Math.abs(sdn2[2] - en2) / Math.max(Math.abs(sdn2[2]), Math.abs(en2), 1)
      const score = (shapeMatch + dimMatch + numMatch) / 3

      if (score > maxScore) {
        maxScore = score
        bestPattern = name
      }
    }

    return { pattern: bestPattern, confidence: maxScore }
  }

  calculateQCCKernel(st1: SpaceTime, st2: SpaceTime) {
    const spatialSep = Math.sqrt((st1.x - st2.x) ** 2 + (st1.y - st2.y) ** 2 + (st1.z - st2.z) ** 2)
    const temporalSep = Math.abs(st1.t - st2.t)
    const lightTravelTime = spatialSep / this.c
    const causalViolation = temporalSep < lightTravelTime

    return {
      spatialSeparation: spatialSep,
      temporalSeparation: temporalSep,
      causalViolation: causalViolation,
      entanglementStrength: causalViolation ? 1.0 : 0.5,
    }
  }

  fullEntanglementAnalysis(p1: Particle, p2: Particle, st1: SpaceTime, st2: SpaceTime): AnalysisResult {
    // SDKP Analysis
    const sdkpStrength = this.entanglementCorrelationStrength(p1, p2)

    // SD&N Analysis
    const sdn1 = [p1.shape || 1.0, p1.dimension || 1, p1.number || 1]
    const sdn2 = [p2.shape || 1.0, p2.dimension || 1, p2.number || 1]
    const sdnAnalysis = this.detectSDNEntanglement(sdn1, sdn2)

    // QCC Analysis
    const qccKernel = this.calculateQCCKernel(st1, st2)

    // Combined confidence
    const combinedConfidence = sdkpStrength * 0.4 + sdnAnalysis.confidence * 0.4 + qccKernel.entanglementStrength * 0.2

    return {
      entangled: combinedConfidence > 0.7,
      confidence: combinedConfidence,
      sdkpStrength: sdkpStrength,
      sdnPattern: sdnAnalysis.pattern,
      sdnConfidence: sdnAnalysis.confidence,
      qccStrength: qccKernel.entanglementStrength,
      spatialSeparation: qccKernel.spatialSeparation,
    }
  }
}

export default function QuantumSimulator() {
  const [simulator] = useState(() => new QuantumSDKPSimulator())
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [consoleOutput, setConsoleOutput] = useState<string[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [stats, setStats] = useState<Record<string, string>>({})

  const addConsoleOutput = (text: string) => {
    setConsoleOutput((prev) => [...prev, text])
  }

  const clearConsole = () => {
    setConsoleOutput([])
  }

  const runBasicDemo = async () => {
    setIsRunning(true)
    clearConsole()
    addConsoleOutput("🚀 Running Basic Entanglement Demo...")

    // Define test particles
    const particle1: Particle = { N: 2.0, S: 1.5, D: 1.0, v: 1e7, shape: 1.0, dimension: 2, number: 1 }
    const particle2: Particle = { N: 2.0, S: 1.5, D: 1.0, v: 1e7, shape: -1.0, dimension: 2, number: 1 }
    const spacetime1: SpaceTime = { x: 0, y: 0, z: 0, t: 0 }
    const spacetime2: SpaceTime = { x: 1000, y: 0, z: 0, t: 1e-6 }

    const analysis = simulator.fullEntanglementAnalysis(particle1, particle2, spacetime1, spacetime2)

    addConsoleOutput(`✅ Entanglement Detected: ${analysis.entangled}`)
    addConsoleOutput(`📊 Overall Confidence: ${analysis.confidence.toFixed(3)}`)
    addConsoleOutput(`🔗 SDKP Correlation: ${analysis.sdkpStrength.toFixed(3)}`)
    addConsoleOutput(`🎯 SD&N Pattern: ${analysis.sdnPattern}`)
    addConsoleOutput(`⚡ QCC Strength: ${analysis.qccStrength.toFixed(3)}`)

    setStats({
      Confidence: analysis.confidence.toFixed(3),
      "SDKP Strength": analysis.sdkpStrength.toFixed(3),
      Pattern: analysis.sdnPattern,
      Entangled: analysis.entangled ? "YES" : "NO",
    })

    setResults([analysis])
    setIsRunning(false)
  }

  const runParameterAnalysis = async () => {
    setIsRunning(true)
    clearConsole()
    addConsoleOutput("🔍 Running Parameter Sensitivity Analysis...")

    const analysisResults: AnalysisResult[] = []
    const velocities = [1e6, 5e6, 1e7, 5e7, 1e8]

    for (let i = 0; i < velocities.length; i++) {
      const v = velocities[i]
      const particle1: Particle = { N: 2.0, S: 1.5, D: 1.0, v: v, shape: 1.0, dimension: 2, number: 1 }
      const particle2: Particle = { N: 2.0, S: 1.5, D: 1.0, v: v, shape: -1.0, dimension: 2, number: 1 }
      const spacetime1: SpaceTime = { x: 0, y: 0, z: 0, t: 0 }
      const spacetime2: SpaceTime = { x: 1000, y: 0, z: 0, t: 1e-6 }

      const analysis = simulator.fullEntanglementAnalysis(particle1, particle2, spacetime1, spacetime2)
      analysisResults.push(analysis)

      addConsoleOutput(`📈 Velocity ${v.toExponential(1)} m/s: Confidence ${analysis.confidence.toFixed(3)}`)
    }

    const avgConfidence = analysisResults.reduce((sum, r) => sum + r.confidence, 0) / analysisResults.length
    const entangledCount = analysisResults.filter((r) => r.entangled).length

    setStats({
      "Avg Confidence": avgConfidence.toFixed(3),
      "Entangled Cases": `${entangledCount}/${analysisResults.length}`,
      "Success Rate": `${((entangledCount / analysisResults.length) * 100).toFixed(1)}%`,
      "Total Tests": analysisResults.length.toString(),
    })

    setResults(analysisResults)
    setIsRunning(false)
  }

  const runScenarioTests = async () => {
    setIsRunning(true)
    clearConsole()
    addConsoleOutput("🧪 Running Multi-Scenario Tests...")

    const scenarios = [
      { name: "Close Proximity", distance: 10 },
      { name: "Medium Range", distance: 1000 },
      { name: "Long Distance", distance: 100000 },
      { name: "Extreme Range", distance: 1000000 },
    ]

    const analysisResults: AnalysisResult[] = []

    for (const scenario of scenarios) {
      const particle1: Particle = { N: 2.0, S: 1.5, D: 1.0, v: 1e7, shape: 1.0, dimension: 2, number: 1 }
      const particle2: Particle = { N: 2.0, S: 1.5, D: 1.0, v: 1e7, shape: -1.0, dimension: 2, number: 1 }
      const spacetime1: SpaceTime = { x: 0, y: 0, z: 0, t: 0 }
      const spacetime2: SpaceTime = { x: scenario.distance, y: 0, z: 0, t: 1e-6 }

      const analysis = simulator.fullEntanglementAnalysis(particle1, particle2, spacetime1, spacetime2)
      analysisResults.push(analysis)

      addConsoleOutput(`🎯 ${scenario.name}: Confidence ${analysis.confidence.toFixed(3)}`)
    }

    const entangledCount = analysisResults.filter((r) => r.entangled).length
    const maxConfidence = Math.max(...analysisResults.map((r) => r.confidence))

    setStats({
      "Scenarios Tested": scenarios.length.toString(),
      Entangled: entangledCount.toString(),
      "Max Confidence": maxConfidence.toFixed(3),
      "Success Rate": `${((entangledCount / scenarios.length) * 100).toFixed(1)}%`,
    })

    setResults(analysisResults)
    setIsRunning(false)
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return "text-green-400"
    if (confidence > 0.5) return "text-yellow-400"
    return "text-red-400"
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 p-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl">
          <h1 className="text-4xl font-bold mb-2 flex items-center justify-center gap-3">
            <Atom className="w-10 h-10" />
            Quantum SDKP Framework Simulations
          </h1>
          <p className="text-lg opacity-90">Shape-Density-Kinematic Principle + SD&N + QCC Integration</p>
        </div>

        {/* Control Panel */}
        <Card className="mb-8 bg-white/10 backdrop-blur-sm border-white/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Rocket className="w-5 h-5" />
              Simulation Control Center
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-4">
              <Button
                onClick={runBasicDemo}
                disabled={isRunning}
                className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
              >
                Run Basic Demo
              </Button>
              <Button
                onClick={runParameterAnalysis}
                disabled={isRunning}
                className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
              >
                Parameter Analysis
              </Button>
              <Button
                onClick={runScenarioTests}
                disabled={isRunning}
                className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
              >
                Multi-Scenario Tests
              </Button>
            </div>
            {isRunning && (
              <div className="mt-4">
                <Progress value={undefined} className="w-full" />
                <p className="text-sm text-center mt-2">Running simulation...</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Console Output */}
        {consoleOutput.length > 0 && (
          <Card className="mb-8 bg-black/50 backdrop-blur-sm border-green-500/30">
            <CardHeader>
              <CardTitle className="text-green-400 font-mono">Console Output</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="font-mono text-sm text-green-300 max-h-64 overflow-y-auto">
                {consoleOutput.map((line, index) => (
                  <div key={index} className="mb-1">
                    {line}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Stats Dashboard */}
        {Object.keys(stats).length > 0 && (
          <Card className="mb-8 bg-white/10 backdrop-blur-sm border-white/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <BarChart3 className="w-5 h-5" />
                Live Results Dashboard
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(stats).map(([label, value]) => (
                  <div key={label} className="bg-gradient-to-r from-blue-500 to-purple-500 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold">{value}</div>
                    <div className="text-sm opacity-90">{label}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {results.length > 0 && (
          <Card className="bg-white/10 backdrop-blur-sm border-white/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Target className="w-5 h-5" />
                Entanglement Analysis Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {results.map((result, index) => (
                  <div key={index} className="bg-white/10 p-6 rounded-lg border border-white/20">
                    <h3 className="text-lg font-semibold mb-4 text-purple-300">Analysis #{index + 1}</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span>Entangled:</span>
                        <Badge variant={result.entangled ? "default" : "destructive"}>
                          {result.entangled ? "YES" : "NO"}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Confidence:</span>
                        <span className={getConfidenceColor(result.confidence)}>{result.confidence.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>SDKP Strength:</span>
                        <span className="text-blue-300">{result.sdkpStrength.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>SD&N Pattern:</span>
                        <span className="text-purple-300">{result.sdnPattern}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>QCC Strength:</span>
                        <span className="text-yellow-300">{result.qccStrength.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Spatial Sep:</span>
                        <span className="text-gray-300">{result.spatialSeparation.toExponential(2)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
