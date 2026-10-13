import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import '../services/auth_service.dart';
import '../services/data_service.dart';
import '../theme.dart';
import 'architecture_tab.dart';
import 'hardware_tab.dart';
import 'overview_tab.dart';
import 'precision_tab.dart';

class HomeShell extends StatefulWidget {
  final User? user;
  const HomeShell({required this.user, super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _tab = 0;
  final _data = DataService();

  static const _titles = ['Overview', 'Precision', 'Architecture', 'Hardware'];

  @override
  void dispose() {
    _data.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: Text(_titles[_tab]),
        actions: [
          IconButton(
            tooltip: widget.user != null ? 'Sign out' : 'Account',
            icon: CircleAvatar(
              radius: 14,
              backgroundColor: AppColors.panel2,
              backgroundImage:
                  widget.user?.photoURL != null ? NetworkImage(widget.user!.photoURL!) : null,
              child: widget.user?.photoURL == null
                  ? const Icon(Icons.person_rounded, size: 16, color: AppColors.muted)
                  : null,
            ),
            onPressed: () async {
              if (widget.user != null) {
                await AuthService.signOut();
              }
            },
          ),
          const SizedBox(width: 6),
        ],
      ),
      body: IndexedStack(
        index: _tab,
        children: [
          OverviewTab(data: _data),
          PrecisionTab(data: _data),
          ArchitectureTab(data: _data),
          HardwareTab(data: _data),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.show_chart_rounded), label: 'Overview'),
          NavigationDestination(icon: Icon(Icons.tune_rounded), label: 'Precision'),
          NavigationDestination(icon: Icon(Icons.device_hub_rounded), label: 'Architecture'),
          NavigationDestination(icon: Icon(Icons.memory_rounded), label: 'Hardware'),
        ],
      ),
    );
  }
}
