"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ExternalLink, Shield, Globe, Coins } from "lucide-react"

export default function LicensingInfo() {
  const licenseTypes = [
    {
      name: "Commercial License",
      description: "For business and commercial applications",
      icon: <Coins className="w-5 h-5" />,
      color: "from-green-500 to-emerald-500",
    },
    {
      name: "Residential License",
      description: "For personal and residential use",
      icon: <Shield className="w-5 h-5" />,
      color: "from-blue-500 to-cyan-500",
    },
    {
      name: "Individual/AI License",
      description: "For individual researchers and AI applications",
      icon: <Globe className="w-5 h-5" />,
      color: "from-purple-500 to-violet-500",
    },
  ]

  return (
    <Card className="bg-white/10 backdrop-blur-sm border-white/20">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-white">
          <Shield className="w-5 h-5" />
          Official NFT Licensing Information
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Official Domain */}
        <div className="text-center p-6 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
          <div className="text-sm opacity-90 mb-2">Official NFT License Domain</div>
          <div className="text-2xl font-mono text-white mb-2">🌐 fathertimesdkp.blockchain</div>
          <Badge variant="outline" className="text-xs">
            via Unstoppable Domains
          </Badge>
          <div className="text-sm opacity-75 mt-3">Canonical reference for all licensing terms and conditions</div>
        </div>

        {/* License Types */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {licenseTypes.map((license, index) => (
            <div key={index} className={`p-4 rounded-lg bg-gradient-to-r ${license.color} text-white`}>
              <div className="flex items-center gap-2 mb-2">
                {license.icon}
                <h3 className="font-semibold">{license.name}</h3>
              </div>
              <p className="text-sm opacity-90">{license.description}</p>
            </div>
          ))}
        </div>

        {/* Domain Usage */}
        <div className="bg-black/20 p-4 rounded-lg border border-white/20">
          <h3 className="text-lg font-semibold mb-3 text-white">Domain Usage</h3>
          <div className="space-y-2 text-sm">
            <div className="flex items-start gap-2">
              <span className="text-green-400">•</span>
              <span>Canonical reference for licensing, terms, and conditions</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-green-400">•</span>
              <span>Applies to all three token types (Commercial, Residential, Individual/AI)</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-green-400">•</span>
              <span>Hosted on Unstoppable Domains blockchain infrastructure</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-green-400">•</span>
              <span>Immutable and decentralized licensing framework</span>
            </div>
          </div>
        </div>

        {/* Access Button */}
        <div className="text-center">
          <Button variant="outline" className="bg-white/10 border-white/30 text-white hover:bg-white/20">
            <ExternalLink className="w-4 h-4 mr-2" />
            Visit License Domain
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
